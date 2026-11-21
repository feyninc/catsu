//! HTTP client with retry, exponential backoff and a per-host circuit breaker.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use llm_resilience::CircuitBreaker;
use rand::Rng;
use reqwest::{Client, RequestBuilder, Response};
use tracing::warn;

use crate::errors::{is_retryable_status, ClientError};

/// Configuration for HTTP requests.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial backoff duration in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds.
    pub max_backoff_ms: u64,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// HTTP/HTTPS proxy URL.
    pub proxy: Option<String>,
    /// PEM-encoded CA certificate.
    pub ca_cert_pem: Option<String>,
    /// Consecutive failures (5xx, 408, network errors) against one provider
    /// host before its circuit opens and requests fail fast. `0` disables
    /// the breaker.
    pub circuit_breaker_threshold: u32,
    /// How long an open circuit rejects requests before a probe is let through.
    pub circuit_breaker_open_secs: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 30_000,
            timeout_secs: 30,
            proxy: None,
            ca_cert_pem: None,
            circuit_breaker_threshold: 5,
            circuit_breaker_open_secs: 30,
        }
    }
}

/// HTTP client wrapper with retry logic.
#[derive(Debug, Clone)]
pub struct HttpClient {
    client: Client,
    config: HttpConfig,
    /// One breaker per `host:port`. The `HttpClient` is cloned into every
    /// provider, so the map is shared and one dead provider cannot trip the
    /// breaker of another.
    breakers: Arc<Mutex<HashMap<String, CircuitBreaker>>>,
}

impl HttpClient {
    /// Create a new HTTP client with the given configuration.
    pub fn new(config: HttpConfig) -> Result<Self, ClientError> {
        let mut builder = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(10);

        if let Some(ref proxy_url) = config.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)?;
            builder = builder.proxy(proxy);
        }

        if let Some(ref pem) = config.ca_cert_pem {
            let cert = reqwest::Certificate::from_pem(pem.as_bytes())?;
            builder = builder.add_root_certificate(cert);
        }

        let client = builder.build()?;

        Ok(Self {
            client,
            config,
            breakers: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get the underlying reqwest client.
    pub fn inner(&self) -> &Client {
        &self.client
    }

    /// The circuit breaker for a provider host, created on first use.
    fn breaker_for(&self, host: &str) -> Option<CircuitBreaker> {
        if self.config.circuit_breaker_threshold == 0 {
            return None;
        }
        let mut breakers = self.breakers.lock().unwrap_or_else(|e| e.into_inner());
        let breaker = breakers.entry(host.to_string()).or_insert_with(|| {
            CircuitBreaker::new(
                self.config.circuit_breaker_threshold as usize,
                1.0,
                Duration::from_secs(self.config.circuit_breaker_open_secs),
            )
        });
        Some(breaker.clone())
    }

    /// Send a request with retry logic.
    ///
    /// The request builder factory is called for each attempt since
    /// RequestBuilder cannot be cloned after body is set.
    ///
    /// Each provider host has a circuit breaker. Once a host has failed
    /// `circuit_breaker_threshold` times in a row, requests to it return
    /// [`ClientError::CircuitOpen`] immediately, including retries that are
    /// still pending in this loop, until `circuit_breaker_open_secs` has
    /// passed and a probe request succeeds.
    pub async fn send_with_retry<F>(&self, request_factory: F) -> Result<Response, ClientError>
    where
        F: Fn(&Client) -> RequestBuilder,
    {
        let mut attempts = 0;
        let mut backoff_ms = self.config.initial_backoff_ms;

        loop {
            attempts += 1;
            let request = request_factory(&self.client).build()?;
            let url = request.url();
            let host = format!(
                "{}:{}",
                url.host_str().unwrap_or_default(),
                url.port_or_known_default().unwrap_or_default()
            );
            let breaker = self.breaker_for(&host);

            // Fail fast while the provider is known to be down, instead of
            // spending the whole retry budget (and its sleeps) on it.
            let permit = match breaker.as_ref() {
                Some(b) => match b.try_acquire() {
                    Some(permit) => Some(permit),
                    None => {
                        return Err(ClientError::CircuitOpen {
                            host,
                            retry_after_ms: b.retry_after_sync().map(|d| d.as_millis() as u64),
                        });
                    }
                },
                None => None,
            };

            match self.client.execute(request).await {
                Ok(response) => {
                    let status = response.status().as_u16();

                    if response.status().is_success() {
                        if let Some(p) = permit {
                            p.success();
                        }
                        return Ok(response);
                    }

                    // Server-side failures count against the provider. A 429
                    // or other 4xx says nothing about its health, so the
                    // permit is dropped without recording an outcome.
                    if let Some(p) = permit {
                        if status >= 500 || status == 408 {
                            p.failure();
                        }
                    }

                    // Check if retryable
                    if is_retryable_status(status) && attempts <= self.config.max_retries {
                        // Extract retry-after header if present
                        let retry_after = response
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<u64>().ok());

                        let wait_ms = if let Some(secs) = retry_after {
                            secs * 1000
                        } else {
                            self.calculate_backoff(backoff_ms)
                        };

                        warn!(
                            status = status,
                            attempt = attempts,
                            wait_ms = wait_ms,
                            "Retryable error, waiting before retry"
                        );

                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                        continue;
                    }

                    // Non-retryable error or max retries exceeded
                    if status == 429 {
                        return Err(ClientError::RateLimited { retry_after: None });
                    }

                    // Try to extract error message from response body
                    let message = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());

                    return Err(ClientError::Api { status, message });
                }
                Err(e) => {
                    if let Some(p) = permit {
                        p.failure();
                    }

                    // Network error
                    if attempts <= self.config.max_retries {
                        let wait_ms = self.calculate_backoff(backoff_ms);

                        warn!(
                            error = %e,
                            attempt = attempts,
                            wait_ms = wait_ms,
                            "Network error, waiting before retry"
                        );

                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                        continue;
                    }

                    if e.is_timeout() {
                        return Err(ClientError::Timeout);
                    }

                    return Err(ClientError::Http(e));
                }
            }
        }
    }

    /// Calculate backoff with jitter.
    fn calculate_backoff(&self, base_ms: u64) -> u64 {
        let mut rng = rand::rng();
        let jitter: f64 = rng.random();
        let jitter_ms = (base_ms as f64 * jitter) as u64;
        base_ms + jitter_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_default_config() {
        let config = HttpConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 100);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.circuit_breaker_threshold, 5);
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(503));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(404));
    }

    #[test]
    fn test_config_with_proxy_and_ca_cert() {
        let config = HttpConfig {
            proxy: Some("http://proxy.example.com:8080".to_string()),
            ca_cert_pem: Some(
                "-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----".to_string(),
            ),
            ..HttpConfig::default()
        };
        assert_eq!(
            config.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );
        assert!(config.ca_cert_pem.is_some());
        assert!(config.ca_cert_pem.unwrap().contains("BEGIN CERTIFICATE"));
    }

    #[test]
    fn test_default_config_has_no_proxy_or_ca_cert() {
        let config = HttpConfig::default();
        assert!(config.proxy.is_none());
        assert!(config.ca_cert_pem.is_none());
    }

    #[test]
    fn test_http_client_new_with_proxy() {
        let mut config = HttpConfig::default();
        config.proxy = Some("http://proxy.example.com:8080".to_string());
        let result = HttpClient::new(config);
        assert!(result.is_ok());
    }

    /// Spawn a local server that answers every request with `status` and
    /// counts how many requests reached it.
    async fn spawn_server(status: u16) -> (String, Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let counter = counter.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = socket.read(&mut buf).await;
                    counter.fetch_add(1, Ordering::SeqCst);
                    let reply = format!(
                        "HTTP/1.1 {status} X\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{{}}"
                    );
                    let _ = socket.write_all(reply.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}/embed"), hits)
    }

    fn fast_config(max_retries: u32, threshold: u32) -> HttpConfig {
        HttpConfig {
            max_retries,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            circuit_breaker_threshold: threshold,
            circuit_breaker_open_secs: 60,
            ..HttpConfig::default()
        }
    }

    #[tokio::test]
    async fn breaker_stops_retry_loop_against_dead_provider() {
        let (url, hits) = spawn_server(503).await;
        let client = HttpClient::new(fast_config(10, 3)).unwrap();

        let err = client
            .send_with_retry(|c| c.post(&url).body("x"))
            .await
            .unwrap_err();

        // Without the breaker this call makes 11 attempts. It now stops
        // after the threshold and reports the open circuit.
        assert!(matches!(err, ClientError::CircuitOpen { .. }), "{err:?}");
        assert_eq!(hits.load(Ordering::SeqCst), 3);

        // Later calls fail fast without touching the network.
        let err = client
            .send_with_retry(|c| c.post(&url).body("x"))
            .await
            .unwrap_err();
        assert!(matches!(err, ClientError::CircuitOpen { .. }));
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn breaker_is_per_provider_host() {
        let (dead, _) = spawn_server(503).await;
        let (healthy, _) = spawn_server(200).await;
        let client = HttpClient::new(fast_config(0, 1)).unwrap();
        // Providers get clones of one HttpClient; a clone must only see the
        // breaker for its own host.
        let other = client.clone();

        assert!(matches!(
            client.send_with_retry(|c| c.post(&dead)).await,
            Err(ClientError::Api { status: 503, .. })
        ));
        assert!(matches!(
            other.send_with_retry(|c| c.post(&dead)).await,
            Err(ClientError::CircuitOpen { .. })
        ));
        assert!(other.send_with_retry(|c| c.post(&healthy)).await.is_ok());
    }

    #[tokio::test]
    async fn client_errors_do_not_trip_breaker() {
        let (url, hits) = spawn_server(400).await;
        let client = HttpClient::new(fast_config(0, 1)).unwrap();
        for _ in 0..3 {
            assert!(matches!(
                client.send_with_retry(|c| c.post(&url)).await,
                Err(ClientError::Api { status: 400, .. })
            ));
        }
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn threshold_zero_disables_breaker() {
        let (url, hits) = spawn_server(503).await;
        let client = HttpClient::new(fast_config(2, 0)).unwrap();
        assert!(matches!(
            client.send_with_retry(|c| c.post(&url)).await,
            Err(ClientError::Api { status: 503, .. })
        ));
        assert_eq!(hits.load(Ordering::SeqCst), 3);
    }
}
