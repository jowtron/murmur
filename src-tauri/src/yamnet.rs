use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::{Arc, Mutex};

const SAMPLE_RATE: f64 = 16000.0;
const CHUNK_SIZE: usize = 15600; // 0.975s at 16kHz — YAMNet's native frame size

// AudioSet class indices for speech and music categories
const SPEECH_CLASSES: &[usize] = &[0, 1, 2, 3, 4, 5, 6]; // Speech, Child speech, Conversation, Narration, Babbling, Speech synth, Shout
const MUSIC_CLASSES: &[usize] = &[
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
    148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeechOnset {
    pub time_secs: f64,
    pub speech_score: f32,
    pub music_score: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrameScores {
    pub time_secs: f64,
    pub speech_score: f32,
    pub music_score: f32,
}

// Opaque TFLite C API types
#[repr(C)]
struct TfLiteModel {
    _private: [u8; 0],
}
#[repr(C)]
struct TfLiteInterpreterOptions {
    _private: [u8; 0],
}
#[repr(C)]
struct TfLiteInterpreter {
    _private: [u8; 0],
}
#[repr(C)]
struct TfLiteTensor {
    _private: [u8; 0],
}

type TfLiteStatus = i32;
const TFLITE_OK: TfLiteStatus = 0;

/// Find the bundled libtensorflowlite_c.dylib
fn find_tflite_lib() -> Result<std::path::PathBuf, String> {
    // Check bundled in app Resources (macOS: AppName.app/Contents/Resources/)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(macos_dir) = exe.parent() {
            let resources = macos_dir.parent().unwrap_or(macos_dir).join("Resources");
            let bundled = resources
                .join("binaries")
                .join("libtensorflowlite_c.dylib");
            if bundled.exists() {
                return Ok(bundled);
            }
            let direct = resources.join("libtensorflowlite_c.dylib");
            if direct.exists() {
                return Ok(direct);
            }
        }
    }
    // Fallback: dev mode
    let dev_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("binaries")
        .join("libtensorflowlite_c.dylib");
    if dev_path.exists() {
        return Ok(dev_path);
    }
    Err("libtensorflowlite_c.dylib not found".to_string())
}

/// Run YAMNet TFLite inference on audio samples and return per-frame speech/music scores
pub fn classify_audio(
    samples: &[f32],
    model_path: &Path,
    progress_cb: Option<Arc<Mutex<dyn FnMut(f32, &str) + Send>>>,
) -> Result<Vec<FrameScores>, String> {
    if let Some(ref cb) = progress_cb {
        if let Ok(mut f) = cb.lock() {
            f(0.01, "Loading YAMNet model...");
        }
    }

    let lib_path = find_tflite_lib()?;

    // Load the TFLite C shared library
    let lib = unsafe { libloading::Library::new(&lib_path) }
        .map_err(|e| format!("Failed to load TFLite library: {}", e))?;

    // Load function pointers
    type ModelCreateFromFile =
        unsafe extern "C" fn(*const c_char) -> *mut TfLiteModel;
    type ModelDelete = unsafe extern "C" fn(*mut TfLiteModel);
    type InterpreterOptionsCreate =
        unsafe extern "C" fn() -> *mut TfLiteInterpreterOptions;
    type InterpreterOptionsDelete =
        unsafe extern "C" fn(*mut TfLiteInterpreterOptions);
    type InterpreterOptionsSetNumThreads =
        unsafe extern "C" fn(*mut TfLiteInterpreterOptions, i32);
    type InterpreterCreate = unsafe extern "C" fn(
        *const TfLiteModel,
        *const TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    type InterpreterDelete = unsafe extern "C" fn(*mut TfLiteInterpreter);
    type InterpreterAllocateTensors =
        unsafe extern "C" fn(*mut TfLiteInterpreter) -> TfLiteStatus;
    type InterpreterInvoke =
        unsafe extern "C" fn(*mut TfLiteInterpreter) -> TfLiteStatus;
    type InterpreterGetInputTensor =
        unsafe extern "C" fn(*const TfLiteInterpreter, i32) -> *mut TfLiteTensor;
    type InterpreterGetOutputTensor =
        unsafe extern "C" fn(*const TfLiteInterpreter, i32) -> *const TfLiteTensor;
    type TensorData =
        unsafe extern "C" fn(*const TfLiteTensor) -> *const std::ffi::c_void;
    type TensorByteSize =
        unsafe extern "C" fn(*const TfLiteTensor) -> usize;

    let model_create: libloading::Symbol<ModelCreateFromFile> = unsafe {
        lib.get(b"TfLiteModelCreateFromFile")
    }
    .map_err(|e| format!("Missing TfLiteModelCreateFromFile: {}", e))?;
    let model_delete: libloading::Symbol<ModelDelete> = unsafe {
        lib.get(b"TfLiteModelDelete")
    }
    .map_err(|e| format!("Missing TfLiteModelDelete: {}", e))?;
    let options_create: libloading::Symbol<InterpreterOptionsCreate> = unsafe {
        lib.get(b"TfLiteInterpreterOptionsCreate")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterOptionsCreate: {}", e))?;
    let options_delete: libloading::Symbol<InterpreterOptionsDelete> = unsafe {
        lib.get(b"TfLiteInterpreterOptionsDelete")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterOptionsDelete: {}", e))?;
    let options_set_threads: libloading::Symbol<InterpreterOptionsSetNumThreads> = unsafe {
        lib.get(b"TfLiteInterpreterOptionsSetNumThreads")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterOptionsSetNumThreads: {}", e))?;
    let interp_create: libloading::Symbol<InterpreterCreate> = unsafe {
        lib.get(b"TfLiteInterpreterCreate")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterCreate: {}", e))?;
    let interp_delete: libloading::Symbol<InterpreterDelete> = unsafe {
        lib.get(b"TfLiteInterpreterDelete")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterDelete: {}", e))?;
    let allocate_tensors: libloading::Symbol<InterpreterAllocateTensors> = unsafe {
        lib.get(b"TfLiteInterpreterAllocateTensors")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterAllocateTensors: {}", e))?;
    let invoke: libloading::Symbol<InterpreterInvoke> = unsafe {
        lib.get(b"TfLiteInterpreterInvoke")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterInvoke: {}", e))?;
    let get_input_tensor: libloading::Symbol<InterpreterGetInputTensor> = unsafe {
        lib.get(b"TfLiteInterpreterGetInputTensor")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterGetInputTensor: {}", e))?;
    let get_output_tensor: libloading::Symbol<InterpreterGetOutputTensor> = unsafe {
        lib.get(b"TfLiteInterpreterGetOutputTensor")
    }
    .map_err(|e| format!("Missing TfLiteInterpreterGetOutputTensor: {}", e))?;
    let tensor_data: libloading::Symbol<TensorData> = unsafe {
        lib.get(b"TfLiteTensorData")
    }
    .map_err(|e| format!("Missing TfLiteTensorData: {}", e))?;
    let tensor_byte_size: libloading::Symbol<TensorByteSize> = unsafe {
        lib.get(b"TfLiteTensorByteSize")
    }
    .map_err(|e| format!("Missing TfLiteTensorByteSize: {}", e))?;

    // Create model from file
    let model_path_c = CString::new(
        model_path
            .to_str()
            .ok_or("Invalid model path encoding")?,
    )
    .map_err(|_| "Model path contains null byte")?;

    let model = unsafe { model_create(model_path_c.as_ptr()) };
    if model.is_null() {
        return Err("Failed to load YAMNet TFLite model".to_string());
    }

    // Create interpreter options
    let options = unsafe { options_create() };
    unsafe { options_set_threads(options, 4) };

    // Create interpreter
    let interpreter = unsafe { interp_create(model, options) };
    if interpreter.is_null() {
        unsafe {
            options_delete(options);
            model_delete(model);
        }
        return Err("Failed to create TFLite interpreter".to_string());
    }

    // Allocate tensors
    let status = unsafe { allocate_tensors(interpreter) };
    if status != TFLITE_OK {
        unsafe {
            interp_delete(interpreter);
            options_delete(options);
            model_delete(model);
        }
        return Err("Failed to allocate TFLite tensors".to_string());
    }

    let total_chunks = if samples.len() >= CHUNK_SIZE {
        (samples.len() - CHUNK_SIZE) / CHUNK_SIZE + 1
    } else {
        0
    };

    if total_chunks == 0 {
        unsafe {
            interp_delete(interpreter);
            options_delete(options);
            model_delete(model);
        }
        return Ok(Vec::new());
    }

    if let Some(ref cb) = progress_cb {
        if let Ok(mut f) = cb.lock() {
            f(0.05, &format!("Classifying {} chunks with YAMNet...", total_chunks));
        }
    }

    let mut frame_scores = Vec::with_capacity(total_chunks);
    let report_interval = (total_chunks / 20).max(1);

    for (idx, start) in (0..samples.len().saturating_sub(CHUNK_SIZE))
        .step_by(CHUNK_SIZE)
        .enumerate()
    {
        if idx % report_interval == 0 {
            if let Some(ref cb) = progress_cb {
                if let Ok(mut f) = cb.lock() {
                    let pct = idx as f32 / total_chunks as f32;
                    f(
                        pct,
                        &format!("Classifying chunk {}/{}...", idx + 1, total_chunks),
                    );
                }
            }
        }

        let chunk = &samples[start..start + CHUNK_SIZE];

        // Copy input data
        let input_tensor = unsafe { get_input_tensor(interpreter, 0) };
        if input_tensor.is_null() {
            continue;
        }
        let input_ptr = unsafe { tensor_data(input_tensor as *const TfLiteTensor) } as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), input_ptr, CHUNK_SIZE);
        }

        // Run inference
        let status = unsafe { invoke(interpreter) };
        if status != TFLITE_OK {
            continue; // skip failed chunks
        }

        // Get output scores
        let output_tensor = unsafe { get_output_tensor(interpreter, 0) };
        if output_tensor.is_null() {
            continue;
        }
        let output_ptr = unsafe { tensor_data(output_tensor) } as *const f32;
        let output_bytes = unsafe { tensor_byte_size(output_tensor) };
        let num_scores = output_bytes / std::mem::size_of::<f32>();
        let scores = unsafe { std::slice::from_raw_parts(output_ptr, num_scores) };

        // Sum speech and music class scores
        let speech_score: f32 = SPEECH_CLASSES
            .iter()
            .filter(|&&i| i < scores.len())
            .map(|&i| scores[i])
            .sum();
        let music_score: f32 = MUSIC_CLASSES
            .iter()
            .filter(|&&i| i < scores.len())
            .map(|&i| scores[i])
            .sum();

        let time_secs = start as f64 / SAMPLE_RATE;

        frame_scores.push(FrameScores {
            time_secs,
            speech_score,
            music_score,
        });
    }

    // Cleanup
    unsafe {
        interp_delete(interpreter);
        options_delete(options);
        model_delete(model);
    }

    if let Some(ref cb) = progress_cb {
        if let Ok(mut f) = cb.lock() {
            f(1.0, &format!("Classified {} frames", frame_scores.len()));
        }
    }

    Ok(frame_scores)
}

/// Find music→speech transitions (speech onsets after music regions)
pub fn find_speech_onsets(frame_scores: &[FrameScores], min_gap_secs: f64) -> Vec<SpeechOnset> {
    if frame_scores.len() < 2 {
        return Vec::new();
    }

    let mut onsets = Vec::new();
    let mut last_onset_time = f64::NEG_INFINITY;

    for i in 1..frame_scores.len() {
        let prev = &frame_scores[i - 1];
        let curr = &frame_scores[i];

        // Detect transition: previous frame was music-dominant, current is speech-dominant
        let prev_is_music = prev.music_score > prev.speech_score;
        let curr_is_speech = curr.speech_score > curr.music_score;

        if prev_is_music && curr_is_speech {
            if curr.time_secs - last_onset_time >= min_gap_secs {
                onsets.push(SpeechOnset {
                    time_secs: curr.time_secs,
                    speech_score: curr.speech_score,
                    music_score: curr.music_score,
                });
                last_onset_time = curr.time_secs;
            }
        }
    }

    onsets
}

/// Find the nearest speech onset to a given time within ±window_secs
pub fn nearest_speech_onset(
    onsets: &[SpeechOnset],
    target_secs: f64,
    window_secs: f64,
) -> Option<f64> {
    onsets
        .iter()
        .filter(|o| (o.time_secs - target_secs).abs() <= window_secs)
        .min_by(|a, b| {
            let da = (a.time_secs - target_secs).abs();
            let db = (b.time_secs - target_secs).abs();
            da.partial_cmp(&db).unwrap()
        })
        .map(|o| o.time_secs)
}
