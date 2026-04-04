mod commands;
mod transcriber;

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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
