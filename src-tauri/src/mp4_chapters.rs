//! Read chapter markers already embedded in an MP4/M4A/M4B container.
//!
//! Audiobooks carry their real chapter list; there is no reason to ask an LLM
//! to guess boundaries from a transcript when the file states them exactly.
//! Two schemes exist in the wild and we handle both:
//!
//! - **QuickTime chapter track.** The audio track carries a `tref/chap` box
//!   naming a second track; that track's samples *are* the titles, timed by its
//!   own `stts` durations. Audible/Libation-style `.m4b` files use this.
//! - **Nero `chpl`.** A flat list in `moov/udta/chpl` with 100 ns timestamps.
//!   ffmpeg and mp4chaps write this one.
//!
//! Symphonia's isomp4 demuxer ignores both, so this is a small hand-rolled box
//! walk. Anything malformed yields an empty list rather than an error: a file
//! without usable chapters is a normal outcome, not a failure.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// One embedded chapter marker: title plus its start in seconds.
#[derive(Debug, Clone)]
pub struct EmbeddedChapter {
    pub title: String,
    pub start_secs: f64,
}

/// A box's payload extent, i.e. the range after its header.
#[derive(Debug, Clone, Copy)]
struct BoxRange {
    start: u64,
    end: u64,
}

type Reader = BufReader<File>;

fn read_u32(r: &mut Reader) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_be_bytes(b))
}

fn read_u64(r: &mut Reader) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_be_bytes(b))
}

/// List the immediate child boxes of `range` as (type, payload range).
fn children(r: &mut Reader, range: BoxRange) -> Vec<([u8; 4], BoxRange)> {
    let mut out = Vec::new();
    let mut pos = range.start;
    // A box needs at least an 8-byte header to be meaningful.
    while pos + 8 <= range.end {
        if r.seek(SeekFrom::Start(pos)).is_err() {
            break;
        }
        let Some(size32) = read_u32(r) else { break };
        let mut btype = [0u8; 4];
        if r.read_exact(&mut btype).is_err() {
            break;
        }
        let (size, header) = match size32 {
            // 1 => the real size follows as a 64-bit value.
            1 => match read_u64(r) {
                Some(s) => (s, 16u64),
                None => break,
            },
            // 0 => the box runs to the end of its parent.
            0 => (range.end - pos, 8),
            n => (n as u64, 8),
        };
        if size < header || pos + size > range.end {
            break;
        }
        let mut payload = BoxRange { start: pos + header, end: pos + size };
        // `meta` carries version+flags ahead of its children.
        if &btype == b"meta" {
            payload.start += 4;
        }
        out.push((btype, payload));
        pos += size;
    }
    out
}

/// Descend a box path (e.g. `["mdia", "minf", "stbl"]`), returning every match.
fn find_path(r: &mut Reader, root: BoxRange, path: &[&[u8; 4]]) -> Vec<BoxRange> {
    let mut current = vec![root];
    for want in path {
        let mut next = Vec::new();
        for range in &current {
            for (btype, payload) in children(r, *range) {
                if &btype == *want {
                    next.push(payload);
                }
            }
        }
        if next.is_empty() {
            return Vec::new();
        }
        current = next;
    }
    current
}

fn find_one(r: &mut Reader, root: BoxRange, path: &[&[u8; 4]]) -> Option<BoxRange> {
    find_path(r, root, path).into_iter().next()
}

/// `tkhd` and `mdhd` are version-dependent: v1 widens the two timestamps from
/// 32 to 64 bits, pushing everything after them 8 bytes further out.
fn header_field(r: &mut Reader, range: BoxRange, v0_offset: u64, v1_offset: u64) -> Option<u32> {
    r.seek(SeekFrom::Start(range.start)).ok()?;
    let version = read_u32(r)? >> 24;
    let offset = if version == 1 { v1_offset } else { v0_offset };
    r.seek(SeekFrom::Start(range.start + offset)).ok()?;
    read_u32(r)
}

fn track_id(r: &mut Reader, trak: BoxRange) -> Option<u32> {
    let tkhd = find_one(r, trak, &[b"tkhd"])?;
    // v0: version/flags(4) creation(4) modification(4) track_id(4)
    // v1: version/flags(4) creation(8) modification(8) track_id(4)
    header_field(r, tkhd, 12, 20)
}

fn media_timescale(r: &mut Reader, trak: BoxRange) -> Option<u32> {
    let mdhd = find_one(r, trak, &[b"mdia", b"mdhd"])?;
    header_field(r, mdhd, 12, 20)
}

/// Read a `stts` box into a flat per-sample duration list.
fn sample_durations(r: &mut Reader, stts: BoxRange) -> Vec<u32> {
    let mut out = Vec::new();
    if r.seek(SeekFrom::Start(stts.start + 4)).is_err() {
        return out;
    }
    let Some(entries) = read_u32(r) else { return out };
    for _ in 0..entries {
        let (Some(count), Some(delta)) = (read_u32(r), read_u32(r)) else { break };
        // Guard against a corrupt count claiming millions of samples.
        let count = count.min(100_000);
        out.extend(std::iter::repeat(delta).take(count as usize));
    }
    out
}

/// Read a `stsz` box into a per-sample byte-size list.
fn sample_sizes(r: &mut Reader, stsz: BoxRange) -> Vec<u32> {
    let mut out = Vec::new();
    if r.seek(SeekFrom::Start(stsz.start + 4)).is_err() {
        return out;
    }
    let (Some(uniform), Some(count)) = (read_u32(r), read_u32(r)) else { return out };
    let count = count.min(100_000) as usize;
    if uniform != 0 {
        return vec![uniform; count];
    }
    for _ in 0..count {
        let Some(size) = read_u32(r) else { break };
        out.push(size);
    }
    out
}

/// Read `stco` (32-bit) or `co64` (64-bit) chunk offsets.
fn chunk_offsets(r: &mut Reader, stbl: BoxRange) -> Vec<u64> {
    let mut out = Vec::new();
    let (range, wide) = match find_one(r, stbl, &[b"stco"]) {
        Some(range) => (range, false),
        None => match find_one(r, stbl, &[b"co64"]) {
            Some(range) => (range, true),
            None => return out,
        },
    };
    if r.seek(SeekFrom::Start(range.start + 4)).is_err() {
        return out;
    }
    let Some(count) = read_u32(r) else { return out };
    for _ in 0..count.min(100_000) {
        let offset = if wide { read_u64(r) } else { read_u32(r).map(u64::from) };
        match offset {
            Some(o) => out.push(o),
            None => break,
        }
    }
    out
}

/// Expand `stsc` into a samples-per-chunk count for each chunk.
fn samples_per_chunk(r: &mut Reader, stsc: BoxRange, chunk_count: usize) -> Vec<u32> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    if r.seek(SeekFrom::Start(stsc.start + 4)).is_err() {
        return vec![1; chunk_count];
    }
    let Some(entries) = read_u32(r) else { return vec![1; chunk_count] };
    for _ in 0..entries.min(100_000) {
        let (Some(first), Some(per), Some(_desc)) = (read_u32(r), read_u32(r), read_u32(r)) else {
            break;
        };
        runs.push((first, per));
    }
    let mut out = vec![1u32; chunk_count];
    for (i, (first, per)) in runs.iter().enumerate() {
        // `first_chunk` is 1-based and runs until the next entry starts.
        let start = first.saturating_sub(1) as usize;
        let end = runs
            .get(i + 1)
            .map(|(next, _)| next.saturating_sub(1) as usize)
            .unwrap_or(chunk_count);
        for slot in out.iter_mut().take(end.min(chunk_count)).skip(start) {
            *slot = *per;
        }
    }
    out
}

/// A chapter-track sample: a 16-bit length followed by the title bytes, then
/// optional atoms (`encd` and friends) we ignore. Some writers omit the length
/// prefix and store bare text, so fall back to treating the payload as UTF-8.
fn decode_title(raw: &[u8]) -> String {
    if raw.len() >= 2 {
        let len = u16::from_be_bytes([raw[0], raw[1]]) as usize;
        if len > 0 && 2 + len <= raw.len() {
            // A UTF-16 BOM can lead the text; String::from_utf8_lossy would
            // mangle it, so decode those explicitly.
            let text = &raw[2..2 + len];
            if text.len() >= 2 && (text[..2] == [0xFE, 0xFF] || text[..2] == [0xFF, 0xFE]) {
                let big_endian = text[..2] == [0xFE, 0xFF];
                let units: Vec<u16> = text[2..]
                    .chunks_exact(2)
                    .map(|c| {
                        if big_endian {
                            u16::from_be_bytes([c[0], c[1]])
                        } else {
                            u16::from_le_bytes([c[0], c[1]])
                        }
                    })
                    .collect();
                return String::from_utf16_lossy(&units).trim().to_string();
            }
            return String::from_utf8_lossy(text).trim().to_string();
        }
    }
    String::from_utf8_lossy(raw).trim().to_string()
}

/// QuickTime chapter track: follow `tref/chap` to the text track and read its
/// samples.
fn read_chapter_track(r: &mut Reader, moov: BoxRange) -> Vec<EmbeddedChapter> {
    let traks = find_path(r, moov, &[b"trak"]);

    // Collect the track ids referenced by any track's `tref/chap`.
    let mut referenced: Vec<u32> = Vec::new();
    for trak in &traks {
        for chap in find_path(r, *trak, &[b"tref", b"chap"]) {
            if r.seek(SeekFrom::Start(chap.start)).is_err() {
                continue;
            }
            let count = (chap.end - chap.start) / 4;
            for _ in 0..count {
                match read_u32(r) {
                    Some(id) => referenced.push(id),
                    None => break,
                }
            }
        }
    }
    if referenced.is_empty() {
        return Vec::new();
    }

    let Some(target) = traks
        .iter()
        .find(|trak| track_id(r, **trak).is_some_and(|id| referenced.contains(&id)))
        .copied()
    else {
        return Vec::new();
    };

    let timescale = media_timescale(r, target).unwrap_or(1000).max(1) as f64;
    let Some(stbl) = find_one(r, target, &[b"mdia", b"minf", b"stbl"]) else {
        return Vec::new();
    };

    let durations = find_one(r, stbl, &[b"stts"])
        .map(|stts| sample_durations(r, stts))
        .unwrap_or_default();
    let sizes = find_one(r, stbl, &[b"stsz"])
        .map(|stsz| sample_sizes(r, stsz))
        .unwrap_or_default();
    let chunks = chunk_offsets(r, stbl);
    if sizes.is_empty() || chunks.is_empty() {
        return Vec::new();
    }
    let per_chunk = find_one(r, stbl, &[b"stsc"])
        .map(|stsc| samples_per_chunk(r, stsc, chunks.len()))
        .unwrap_or_else(|| vec![1; chunks.len()]);

    // Walk chunks to get each sample's absolute file offset.
    let mut offsets: Vec<u64> = Vec::with_capacity(sizes.len());
    let mut sample = 0usize;
    for (chunk_index, chunk_start) in chunks.iter().enumerate() {
        let mut offset = *chunk_start;
        let count = per_chunk.get(chunk_index).copied().unwrap_or(1);
        for _ in 0..count {
            if sample >= sizes.len() {
                break;
            }
            offsets.push(offset);
            offset += sizes[sample] as u64;
            sample += 1;
        }
    }

    let mut chapters = Vec::with_capacity(offsets.len());
    let mut ticks: u64 = 0;
    for (i, offset) in offsets.iter().enumerate() {
        let size = sizes[i] as usize;
        // Titles are short; a huge size means we mis-parsed the tables.
        if size > 0 && size <= 8192 && r.seek(SeekFrom::Start(*offset)).is_ok() {
            let mut raw = vec![0u8; size];
            if r.read_exact(&mut raw).is_ok() {
                let title = decode_title(&raw);
                if !title.is_empty() {
                    chapters.push(EmbeddedChapter {
                        title,
                        start_secs: ticks as f64 / timescale,
                    });
                }
            }
        }
        ticks += durations.get(i).copied().unwrap_or(0) as u64;
    }
    chapters
}

/// Nero `chpl`: version/flags, a reserved byte, a 32-bit count, then per entry
/// a 64-bit start in 100 ns units and a length-prefixed UTF-8 title.
fn read_nero_chpl(r: &mut Reader, moov: BoxRange) -> Vec<EmbeddedChapter> {
    let mut chapters = Vec::new();
    let Some(chpl) = find_one(r, moov, &[b"udta", b"chpl"]) else {
        return chapters;
    };
    if r.seek(SeekFrom::Start(chpl.start)).is_err() {
        return chapters;
    }
    let Some(version_flags) = read_u32(r) else { return chapters };
    // v1 inserts a reserved byte before the count; v0 uses a single-byte count.
    let count = if version_flags >> 24 == 1 {
        let mut skip = [0u8; 1];
        if r.read_exact(&mut skip).is_err() {
            return chapters;
        }
        match read_u32(r) {
            Some(c) => c,
            None => return chapters,
        }
    } else {
        let mut byte = [0u8; 1];
        if r.read_exact(&mut byte).is_err() {
            return chapters;
        }
        byte[0] as u32
    };

    for _ in 0..count.min(100_000) {
        let Some(start_100ns) = read_u64(r) else { break };
        let mut len = [0u8; 1];
        if r.read_exact(&mut len).is_err() {
            break;
        }
        let mut raw = vec![0u8; len[0] as usize];
        if r.read_exact(&mut raw).is_err() {
            break;
        }
        let title = String::from_utf8_lossy(&raw).trim().to_string();
        if !title.is_empty() {
            chapters.push(EmbeddedChapter {
                title,
                start_secs: start_100ns as f64 / 10_000_000.0,
            });
        }
    }
    chapters
}

/// Read embedded chapters from an MP4-family file. Returns an empty list when
/// the container has none, is not MP4 at all, or is too damaged to parse.
pub fn read_chapters(path: &Path) -> Vec<EmbeddedChapter> {
    let Ok(file) = File::open(path) else { return Vec::new() };
    let Ok(size) = file.metadata().map(|m| m.len()) else { return Vec::new() };
    let mut reader = BufReader::new(file);
    let root = BoxRange { start: 0, end: size };

    let Some(moov) = find_one(&mut reader, root, &[b"moov"]) else {
        return Vec::new();
    };

    // Prefer the chapter track: when both are present it is the one players
    // honour, and `chpl` is often a lossy duplicate of it.
    let mut chapters = read_chapter_track(&mut reader, moov);
    if chapters.is_empty() {
        chapters = read_nero_chpl(&mut reader, moov);
    }

    chapters.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
    chapters.dedup_by(|a, b| (a.start_secs - b.start_secs).abs() < 0.001 && a.title == b.title);
    chapters
}

#[cfg(test)]
mod tests {
    use super::decode_title;

    #[test]
    fn reads_length_prefixed_utf8() {
        let mut raw = vec![0x00, 0x0B];
        raw.extend(b"1. Awakening"[..11].iter());
        assert_eq!(decode_title(&raw), "1. Awakening"[..11].to_string());
    }

    #[test]
    fn reads_utf16_with_bom() {
        let text = "Epigraph";
        let mut body = vec![0xFE, 0xFF];
        for unit in text.encode_utf16() {
            body.extend(unit.to_be_bytes());
        }
        let mut raw = (body.len() as u16).to_be_bytes().to_vec();
        raw.extend(&body);
        assert_eq!(decode_title(&raw), "Epigraph");
    }

    #[test]
    fn ignores_trailing_atoms_after_the_title() {
        // Real chapter samples append boxes like `encd` after the text.
        let mut raw = vec![0x00, 0x05];
        raw.extend(b"Intro");
        raw.extend(b"\x00\x00\x00\x0Cencd\x00\x00\x01\x00");
        assert_eq!(decode_title(&raw), "Intro");
    }

    #[test]
    fn falls_back_when_there_is_no_length_prefix() {
        assert_eq!(decode_title(b"Bare Title"), "Bare Title");
    }

    /// Build the smallest file that exercises the Nero path: ftyp + moov/udta/chpl.
    fn write_chpl_file(entries: &[(u64, &str)]) -> std::path::PathBuf {
        fn boxed(kind: &[u8; 4], payload: Vec<u8>) -> Vec<u8> {
            let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
            out.extend(kind);
            out.extend(payload);
            out
        }

        let mut chpl = vec![0x01, 0, 0, 0]; // version 1, no flags
        chpl.push(0); // reserved
        chpl.extend((entries.len() as u32).to_be_bytes());
        for (start_100ns, title) in entries {
            chpl.extend(start_100ns.to_be_bytes());
            chpl.push(title.len() as u8);
            chpl.extend(title.as_bytes());
        }

        let udta = boxed(b"udta", boxed(b"chpl", chpl));
        let moov = boxed(b"moov", udta);
        let mut file = boxed(b"ftyp", b"M4B mp42".to_vec());
        file.extend(moov);

        let path = std::env::temp_dir().join(format!("murmur_chpl_{}.m4b", entries.len()));
        std::fs::write(&path, file).unwrap();
        path
    }

    #[test]
    fn reads_nero_chpl_when_there_is_no_chapter_track() {
        let path = write_chpl_file(&[
            (0, "Opening Credits"),
            (150_000_000, "Chapter One"),
            (900_000_000, "Chapter Two"),
        ]);
        let chapters = super::read_chapters(&path);
        std::fs::remove_file(&path).ok();

        let got: Vec<(String, f64)> =
            chapters.into_iter().map(|c| (c.title, c.start_secs)).collect();
        assert_eq!(
            got,
            vec![
                ("Opening Credits".to_string(), 0.0),
                ("Chapter One".to_string(), 15.0),
                ("Chapter Two".to_string(), 90.0),
            ]
        );
    }

    #[test]
    fn returns_nothing_for_a_file_that_is_not_mp4() {
        let path = std::env::temp_dir().join("murmur_not_mp4.m4b");
        std::fs::write(&path, b"fLaC not really an mp4 at all").unwrap();
        let chapters = super::read_chapters(&path);
        std::fs::remove_file(&path).ok();
        assert!(chapters.is_empty());
    }
}
