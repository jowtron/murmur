use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Build a curl `-K -` config snippet carrying an Authorization header.
/// Feeding the header via stdin keeps the API key out of the process list
/// (`ps` shows curl's argv to every local process).
pub fn auth_header_config(value: &str) -> String {
    // curl config quoting: backslash-escape the only two characters that can
    // break out of a double-quoted value.
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("header = \"Authorization: {}\"\n", escaped)
}

/// Run curl with `stdin_config` piped to `-K -` (for secrets) plus `args`.
/// If `cancel` fires while curl is running, the child process is killed and
/// `Err("Cancelled")` is returned. Returns stdout on success. Callers that
/// want HTTP error statuses to fail should pass `--fail-with-body` in `args`;
/// without it curl exits 0 and the error body comes back as stdout.
pub async fn run_curl(
    args: &[&str],
    stdin_config: &str,
    cancel: Option<&CancellationToken>,
) -> Result<Vec<u8>, String> {
    let mut child = Command::new("curl")
        .args(["-sS", "-K", "-"])
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("curl failed to start: {}", e))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to open curl stdin".to_string())?;
    stdin
        .write_all(stdin_config.as_bytes())
        .await
        .map_err(|e| format!("Failed to write curl config: {}", e))?;
    drop(stdin); // close stdin so curl stops reading config

    let wait = child.wait_with_output();
    let output = match cancel {
        Some(token) => tokio::select! {
            // Dropping the wait future drops the child; kill_on_drop reaps curl.
            _ = token.cancelled() => return Err("Cancelled".to_string()),
            out = wait => out,
        },
        None => wait.await,
    }
    .map_err(|e| format!("curl failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let body = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "curl exited {}: {} {}",
            output.status,
            stderr.trim(),
            body.trim()
        ));
    }
    Ok(output.stdout)
}
