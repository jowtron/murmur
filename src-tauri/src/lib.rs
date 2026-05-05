mod assemblyai;
mod commands;
mod deepgram;
mod flac_utils;
mod sherpa;
mod transcriber;
mod gap_detection;
mod template;
mod yamnet;

use commands::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new(2);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::is_model_ready,
            commands::download_model,
            commands::get_models_dir,
            commands::get_gpu_info,
            commands::set_concurrency,
            commands::cancel_job,
            commands::transcribe_file,
            commands::detect_chapters,
            commands::scan_directory,
            commands::get_audio_duration,
            commands::read_text_file,
            commands::write_text_file,
            commands::file_exists,
            commands::detect_energy_gaps,
            commands::detect_chapters_with_gaps,
            commands::check_seekability,
            commands::fix_seektable,
            commands::fix_seektables_batch,
            commands::embed_chapters_in_flac,
            commands::write_cue_file,
            commands::delete_file,
            commands::check_transcription_exists,
            commands::get_waveform_peaks,
            commands::get_audio_region_pcm,
            commands::save_audio_template,
            commands::list_audio_templates,
            commands::get_template_pcm,
            commands::delete_audio_template,
            commands::find_template_matches,
            commands::download_podcast_episode,
            commands::transcribe_assemblyai,
            commands::check_diarization_exists,
            commands::transcribe_deepgram,
            commands::transcribe_sherpa,
            commands::sherpa_models_ready,
            commands::sherpa_models_dir,
            commands::download_sherpa_models,
            commands::download_sherpa_model,
            commands::list_sherpa_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
