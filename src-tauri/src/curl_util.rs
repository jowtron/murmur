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

/// One reading of curl's progress meter during an upload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UploadProgress {
    /// Bytes sent so far (the meter's resolution: ~1 MB once past 100 MB).
    pub sent: u64,
    /// Current transfer speed in bytes per second.
    pub bytes_per_sec: u64,
    /// The meter's "% Xferd" reached 100: every byte is sent, and curl is now
    /// waiting on the response. `sent` alone can't say this, being rounded.
    pub done: bool,
}

/// `run_curl`, but with curl's progress meter left on and parsed as it runs,
/// calling `on_progress` roughly once a second. For uploads, where a big file
/// over a slow link otherwise sits silent for minutes.
///
/// The meter goes to stderr, redrawn in place with `\r`. Lines that aren't
/// meter rows (curl's own errors, retry notices) are kept for the error text.
/// A `--retry` restarts the meter, so `sent` can drop back towards zero;
/// that is reported as-is rather than hidden.
pub async fn run_curl_with_progress(
    args: &[&str],
    stdin_config: &str,
    cancel: Option<&CancellationToken>,
    mut on_progress: impl FnMut(UploadProgress),
) -> Result<Vec<u8>, String> {
    use tokio::io::AsyncReadExt;

    // -S without -s: errors still print, and so does the meter.
    let mut child = Command::new("curl")
        .args(["-S", "-K", "-"])
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
    drop(stdin);

    let mut stdout = child.stdout.take().ok_or("Failed to open curl stdout")?;
    let mut stderr = child.stderr.take().ok_or("Failed to open curl stderr")?;

    let work = async {
        // Drain stdout alongside stderr so a large response body can't fill
        // the pipe and stall curl.
        let read_out = async {
            let mut v = Vec::new();
            stdout.read_to_end(&mut v).await.map(|_| v)
        };
        let read_err = async {
            let mut errors = String::new();
            let mut line: Vec<u8> = Vec::new();
            let mut buf = [0u8; 4096];
            let mut take_line = |line: &mut Vec<u8>, errors: &mut String| {
                let text = String::from_utf8_lossy(line).to_string();
                line.clear();
                if let Some(p) = parse_meter_line(&text) {
                    on_progress(p);
                } else if !is_meter_header(&text) {
                    errors.push_str(text.trim());
                    errors.push(' ');
                }
            };
            loop {
                let n = stderr.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                for &b in &buf[..n] {
                    if b == b'\r' || b == b'\n' {
                        take_line(&mut line, &mut errors);
                    } else {
                        line.push(b);
                    }
                }
            }
            take_line(&mut line, &mut errors);
            Ok::<String, std::io::Error>(errors)
        };
        let (out, err) = tokio::join!(read_out, read_err);
        let status = child.wait().await?;
        Ok::<_, std::io::Error>((status, out?, err?))
    };

    let (status, out, errors) = match cancel {
        Some(token) => tokio::select! {
            _ = token.cancelled() => return Err("Cancelled".to_string()),
            r = work => r,
        },
        None => work.await,
    }
    .map_err(|e| format!("curl failed: {}", e))?;

    if !status.success() {
        let body = String::from_utf8_lossy(&out);
        return Err(format!(
            "curl exited {}: {} {}",
            status,
            errors.trim(),
            body.trim()
        ));
    }
    Ok(out)
}

/// The two header rows curl prints above its meter.
fn is_meter_header(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with("% Total") || t.starts_with("Dload")
}

/// Parse one row of curl's default progress meter:
///
/// ```text
///  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
///                                 Dload  Upload   Total   Spent    Left  Speed
///  66  158M    0     0   66  105M      0   312k  0:08:40  0:05:46  0:02:54  322k
/// ```
///
/// Column 5 is the percent sent, 6 the bytes sent, 12 the current speed.
fn parse_meter_line(line: &str) -> Option<UploadProgress> {
    let cols: Vec<&str> = line.split_whitespace().collect();
    if cols.len() != 12 || cols[0].parse::<u32>().is_err() {
        return None;
    }
    Some(UploadProgress {
        sent: parse_meter_size(cols[5])?,
        bytes_per_sec: parse_meter_size(cols[11])?,
        done: cols[4] == "100",
    })
}

/// curl's meter sizes: plain bytes, or a k/M/G/T/P suffix in powers of 1024,
/// with or without a decimal ("5859k", "105M", "1.2G").
fn parse_meter_size(s: &str) -> Option<u64> {
    let (num, mult) = match s.chars().last()? {
        'k' => (&s[..s.len() - 1], 1u64 << 10),
        'M' => (&s[..s.len() - 1], 1u64 << 20),
        'G' => (&s[..s.len() - 1], 1u64 << 30),
        'T' => (&s[..s.len() - 1], 1u64 << 40),
        'P' => (&s[..s.len() - 1], 1u64 << 50),
        _ => (s, 1),
    };
    let v: f64 = num.parse().ok()?;
    Some((v * mult as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meter_rows() {
        assert_eq!(
            parse_meter_line(" 66  158M    0     0   66  105M      0   312k  0:08:40  0:05:46  0:02:54  322k"),
            Some(UploadProgress { sent: 105 << 20, bytes_per_sec: 322 << 10, done: false })
        );
        assert_eq!(
            parse_meter_line("  0 5859k    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0"),
            Some(UploadProgress { sent: 0, bytes_per_sec: 0, done: false })
        );
        assert!(parse_meter_line("100  158M    0     0  100  158M      0   312k  0:08:40  0:08:40 --:--:--     0").unwrap().done);
        assert_eq!(parse_meter_size("1.5G"), Some(3 << 29));
        assert_eq!(parse_meter_size("812"), Some(812));
        assert!(parse_meter_line("  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current").is_none());
        assert!(parse_meter_line("curl: (52) Empty reply from server").is_none());
        assert!(is_meter_header("                                 Dload  Upload   Total   Spent    Left  Speed"));
    }

    /// Real curl against a throttled local listener: progress must arrive
    /// while the upload is still running, and climb.
    #[tokio::test]
    async fn live_upload_reports_progress() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 65536];
            let mut total = 0usize;
            while total < 3_000_000 {
                match sock.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => total += n,
                }
            }
            let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
        });
        let dir = std::env::temp_dir().join(format!("murmur_curl_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("blob.bin");
        std::fs::write(&file, vec![7u8; 3_000_000]).unwrap();
        let data = format!("@{}", file.display());
        let url = format!("http://127.0.0.1:{}/", port);
        let mut seen = Vec::new();
        let out = run_curl_with_progress(
            &["-X", "POST", "--data-binary", &data, "--limit-rate", "1M", &url],
            "",
            None,
            |p| seen.push(p),
        )
        .await
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(out, b"ok");
        let mid = seen.iter().filter(|p| p.sent > 0 && p.sent < 3_000_000).count();
        assert!(mid >= 1, "no mid-upload progress: {:?}", seen);
        assert!(seen.iter().any(|p| p.bytes_per_sec > 0), "no speed: {:?}", seen);
        assert!(seen.last().unwrap().done, "never reported done: {:?}", seen);
    }
}
