use std::io::Read;
use std::path::Path;

/// Find the metaflac binary — checks bundled Resources first, then common paths.
fn find_metaflac() -> Result<std::path::PathBuf, String> {
    // Check bundled in app Resources (macOS: AppName.app/Contents/Resources/)
    if let Ok(exe) = std::env::current_exe() {
        // exe is at AppName.app/Contents/MacOS/appname
        if let Some(macos_dir) = exe.parent() {
            let resources = macos_dir.parent().unwrap_or(macos_dir).join("Resources");
            let bundled = resources.join("binaries").join("metaflac");
            if bundled.exists() {
                return Ok(bundled);
            }
            // Also check directly in Resources
            let bundled = resources.join("metaflac");
            if bundled.exists() {
                return Ok(bundled);
            }
        }
    }

    // Fall back to system paths
    let candidates = [
        "/opt/homebrew/bin/metaflac",
        "/usr/local/bin/metaflac",
        "/usr/bin/metaflac",
    ];
    for candidate in &candidates {
        let p = std::path::PathBuf::from(candidate);
        if p.exists() {
            return Ok(p);
        }
    }
    Err("metaflac not found. Install with: brew install flac".to_string())
}

/// Create a Command for metaflac with the correct library path set.
fn metaflac_command() -> Result<std::process::Command, String> {
    let path = find_metaflac()?;
    let mut cmd = std::process::Command::new(&path);
    // Set DYLD_LIBRARY_PATH so bundled dylibs are found
    if let Some(dir) = path.parent() {
        cmd.env("DYLD_LIBRARY_PATH", dir);
    }
    Ok(cmd)
}

/// Check if a FLAC file contains a SEEKTABLE metadata block.
/// Returns Ok(true) if seektable present, Ok(false) if missing, Err if not a FLAC or unreadable.
pub fn has_seektable(path: &Path) -> Result<bool, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open file: {}", e))?;

    // Read magic bytes
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| format!("Failed to read FLAC header: {}", e))?;

    if &magic != b"fLaC" {
        return Err("Not a FLAC file".to_string());
    }

    // Read metadata blocks
    loop {
        let mut header = [0u8; 4];
        file.read_exact(&mut header)
            .map_err(|e| format!("Failed to read metadata block header: {}", e))?;

        let is_last = (header[0] & 0x80) != 0;
        let block_type = header[0] & 0x7F;
        let block_length = ((header[1] as u32) << 16)
            | ((header[2] as u32) << 8)
            | (header[3] as u32);

        // Block type 3 = SEEKTABLE
        if block_type == 3 {
            return Ok(true);
        }

        // Skip this block's data
        let mut skip = vec![0u8; block_length as usize];
        file.read_exact(&mut skip)
            .map_err(|e| format!("Failed to skip block data: {}", e))?;

        if is_last {
            break;
        }
    }

    Ok(false)
}

/// Add seek points to a FLAC file using the metaflac CLI tool.
/// Adds one seek point per second.
pub fn add_seektable(path: &Path) -> Result<(), String> {
    let status = metaflac_command()?
        .args(["--add-seekpoint=1s", path.to_str().ok_or("Invalid path")?])
        .status()
        .map_err(|e| format!("Failed to run metaflac: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err("metaflac failed to add seek points".to_string())
    }
}

/// Check a file and return info about its seek capability
pub fn check_seekability(path: &Path) -> SeekInfo {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "flac" => match has_seektable(path) {
            Ok(true) => SeekInfo {
                seekable: true,
                has_seektable: true,
                fixable: false,
                format: "FLAC".to_string(),
                message: None,
            },
            Ok(false) => SeekInfo {
                seekable: false,
                has_seektable: false,
                fixable: true,
                format: "FLAC".to_string(),
                message: Some(
                    "Missing seek table — seeking will be inaccurate in media players"
                        .to_string(),
                ),
            },
            Err(e) => SeekInfo {
                seekable: false,
                has_seektable: false,
                fixable: false,
                format: "FLAC".to_string(),
                message: Some(format!("Could not read FLAC metadata: {}", e)),
            },
        },
        "wav" | "aiff" | "aif" => SeekInfo {
            seekable: true,
            has_seektable: true,
            fixable: false,
            format: ext.to_uppercase(),
            message: None,
        },
        _ => SeekInfo {
            seekable: true,
            has_seektable: true,
            fixable: false,
            format: ext.to_uppercase(),
            message: None,
        },
    }
}

/// Embed chapters as Vorbis comment tags in a FLAC file.
/// Uses CHAPTERxxx=HH:MM:SS.mmm and CHAPTERxxxNAME=title format.
pub fn embed_chapters(
    path: &Path,
    chapters: &[(String, f64)], // (title, start_secs)
) -> Result<(), String> {
    let metaflac = find_metaflac()?;
    let metaflac_dir = metaflac.parent().map(|p| p.to_path_buf());

    let mut run = |args: &[&str]| -> Result<(), String> {
        let mut cmd = std::process::Command::new(&metaflac);
        if let Some(ref dir) = metaflac_dir {
            cmd.env("DYLD_LIBRARY_PATH", dir);
        }
        cmd.args(args).status().map_err(|e| format!("metaflac error: {}", e))?;
        Ok(())
    };

    // First remove any existing chapter tags
    let path_str = path.to_str().ok_or("Invalid path")?;
    let _ = run(&["--remove-tag=CHAPTER", path_str]);

    for i in 0..chapters.len() + 10 {
        let tag = format!("--remove-tag=CHAPTER{:03}", i + 1);
        let name_tag = format!("--remove-tag=CHAPTER{:03}NAME", i + 1);
        let _ = run(&[&tag, path_str]);
        let _ = run(&[&name_tag, path_str]);
    }

    // Set new chapter tags
    for (i, (title, secs)) in chapters.iter().enumerate() {
        let h = (*secs / 3600.0) as u32;
        let m = ((*secs % 3600.0) / 60.0) as u32;
        let s = (*secs % 60.0) as u32;
        let ms = ((*secs % 1.0) * 1000.0) as u32;
        let timestamp = format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms);

        let tag = format!("CHAPTER{:03}={}", i + 1, timestamp);
        let name_tag = format!("CHAPTER{:03}NAME={}", i + 1, title);

        let status = run(&["--set-tag", &tag, path_str]);
        if status.is_err() {
            return Err(format!("Failed to set tag: {}", tag));
        }

        let status = run(&["--set-tag", &name_tag, path_str]);
        if status.is_err() {
            return Err(format!("Failed to set tag: {}", name_tag));
        }
    }

    Ok(())
}

/// Generate a .cue file for an audio file with chapters.
pub fn generate_cue(
    audio_path: &Path,
    chapters: &[(String, f64)], // (title, start_secs)
) -> Result<String, String> {
    let audio_filename = audio_path
        .file_name()
        .ok_or("Invalid audio path")?
        .to_string_lossy();

    let ext = audio_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav")
        .to_uppercase();

    let file_type = match ext.as_str() {
        "FLAC" => "FLAC",
        "WAV" | "AIFF" | "AIF" => "WAVE",
        "MP3" => "MP3",
        _ => "WAVE",
    };

    let mut cue = String::new();
    cue.push_str(&format!("FILE \"{}\" {}\n", audio_filename, file_type));

    for (i, (title, secs)) in chapters.iter().enumerate() {
        let m = (*secs / 60.0) as u32;
        let s = (*secs % 60.0) as u32;
        let f = ((*secs % 1.0) * 75.0) as u32; // CUE uses 75 frames/sec

        cue.push_str(&format!("  TRACK {:02} AUDIO\n", i + 1));
        cue.push_str(&format!("    TITLE \"{}\"\n", title));
        cue.push_str(&format!("    INDEX 01 {:02}:{:02}:{:02}\n", m, s, f));
    }

    Ok(cue)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SeekInfo {
    pub seekable: bool,
    pub has_seektable: bool,
    pub fixable: bool,
    pub format: String,
    pub message: Option<String>,
}
