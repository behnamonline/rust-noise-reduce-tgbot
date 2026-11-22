use std::{
    env,
    error::Error,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use nnnoiseless::DenoiseState;
use teloxide::{
    net::Download,
    prelude::*,
    types::{ChatAction, InputFile, Update, UpdateKind},
};
use tokio::{fs, process::Command};

type BoxErr = Box<dyn Error + Send + Sync>;

const DEFAULT_PORT: &str = "8585";
// RNNoise works at 48 kHz mono (same as Telegram's Opus voice notes)
const SAMPLE_RATE: &str = "48000";
// Reject very large files (bytes)
const MAX_FILE_SIZE: u32 = 10_000_000;

#[derive(Clone)]
struct AppState {
    bot: Bot,
    webhook_secret: Arc<str>,
    webhook_set: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Webhook plumbing (same idea as before)
// ---------------------------------------------------------------------------

async fn ensure_webhook(state: &AppState, host: &str) {
    if state.webhook_set.load(Ordering::SeqCst) {
        return;
    }
    let url = format!("https://{host}/webhook");
    let parsed = match url.parse() {
        Ok(u) => u,
        Err(e) => {
            log::error!("Invalid webhook URL {url}: {e}");
            return;
        }
    };
    match state
        .bot
        .set_webhook(parsed)
        .secret_token(state.webhook_secret.to_string())
        .await
    {
        Ok(_) => {
            state.webhook_set.store(true, Ordering::SeqCst);
            log::info!("Webhook set to {url}");
        }
        Err(e) => log::error!("Failed to set webhook: {e}"),
    }
}

async fn index(State(state): State<AppState>, headers: HeaderMap) -> &'static str {
    let host = headers
        .get("x-forwarded-host")
        .or_else(|| headers.get("host"))
        .and_then(|v| v.to_str().ok());
    if state.webhook_set.load(Ordering::SeqCst) {
        return "Bot is running, webhook is set.";
    }
    if let Some(host) = host {
        let state = state.clone();
        let host = host.to_string();
        tokio::spawn(async move {
            ensure_webhook(&state, &host).await;
        });
    }
    "Setting webhook... refresh this page in a few seconds."
}

async fn webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(update): Json<Update>,
) -> StatusCode {
    let secret_ok = headers
        .get("x-telegram-bot-api-secret-token")
        .and_then(|v| v.to_str().ok())
        == Some(&*state.webhook_secret);
    if !secret_ok {
        return StatusCode::UNAUTHORIZED;
    }

    if let UpdateKind::Message(msg) = update.kind {
        let bot = state.bot.clone();
        let chat_id = msg.chat.id;

        if let Some(voice) = msg.voice() {
            if voice.file.size > MAX_FILE_SIZE {
                tokio::spawn(async move {
                    let _ = bot.send_message(chat_id, "فایل صوتی خیلی بزرگه 😕").await;
                });
            } else {
                let file_id = voice.file.id.clone();
                // Process in the background so Telegram gets its 200 immediately
                tokio::spawn(async move {
                    if let Err(e) = handle_voice(&bot, chat_id, file_id).await {
                        log::error!("handle_voice failed: {e}");
                        let _ = bot
                            .send_message(chat_id, "متأسفانه پردازش ویس با خطا مواجه شد 😕")
                            .await;
                    }
                });
            }
        } else if msg.text().is_some() {
            tokio::spawn(async move {
                let _ = bot
                    .send_message(chat_id, "یک ویس برام بفرست تا نویزش رو بگیرم 🎙️")
                    .await;
            });
        }
    }
    StatusCode::OK
}

// ---------------------------------------------------------------------------
// Voice handling
// ---------------------------------------------------------------------------

async fn handle_voice(bot: &Bot, chat_id: ChatId, file_id: String) -> Result<(), BoxErr> {
    let _ = bot.send_chat_action(chat_id, ChatAction::RecordVoice).await;

    // 1. Download the OGG/Opus voice note
    let file = bot.get_file(file_id).await?;
    let mut ogg_in: Vec<u8> = Vec::new();
    bot.download_file(&file.path, &mut ogg_in).await?;

    // 2. Decode -> denoise -> encode
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = env::temp_dir().join(format!("denoise-{}-{nanos}", chat_id.0));
    fs::create_dir_all(&dir).await?;
    let result = run_pipeline(&dir, &ogg_in).await;
    let _ = fs::remove_dir_all(&dir).await;
    let ogg_out = result?;

    // 3. Send back
    bot.send_voice(chat_id, InputFile::memory(ogg_out).file_name("clean.ogg"))
        .await?;
    Ok(())
}

async fn run_pipeline(dir: &Path, ogg_in: &[u8]) -> Result<Vec<u8>, BoxErr> {
    let in_ogg = dir.join("in.ogg");
    let in_raw = dir.join("in.raw");
    let out_raw = dir.join("out.raw");
    let out_ogg = dir.join("out.ogg");

    fs::write(&in_ogg, ogg_in).await?;

    // Opus -> 48 kHz mono signed 16-bit PCM
    ffmpeg(&[
        "-i", path_str(&in_ogg)?,
        "-ac", "1",
        "-ar", SAMPLE_RATE,
        "-f", "s16le",
        path_str(&in_raw)?,
    ])
    .await?;

    let raw = fs::read(&in_raw).await?;
    let samples: Vec<f32> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
        .collect();

    // CPU-bound work: keep it off the async executor threads
    let cleaned = tokio::task::spawn_blocking(move || denoise(&samples)).await?;

    let mut bytes = Vec::with_capacity(cleaned.len() * 2);
    for s in cleaned {
        bytes.extend_from_slice(&(s.round().clamp(-32768.0, 32767.0) as i16).to_le_bytes());
    }
    fs::write(&out_raw, bytes).await?;

    // PCM -> OGG/Opus (the format Telegram expects for voice messages)
    ffmpeg(&[
        "-f", "s16le",
        "-ar", SAMPLE_RATE,
        "-ac", "1",
        "-i", path_str(&out_raw)?,
        "-c:a", "libopus",
        "-b:a", "32k",
        "-application", "voip",
        path_str(&out_ogg)?,
    ])
    .await?;

    Ok(fs::read(&out_ogg).await?)
}

/// RNNoise denoising. Input/output samples are in i16 range (as f32).
fn denoise(samples: &[f32]) -> Vec<f32> {
    const N: usize = DenoiseState::FRAME_SIZE; // 480 samples = 10 ms
    let mut state = DenoiseState::new();
    let mut out = Vec::with_capacity(samples.len() + N);
    let mut out_frame = [0.0f32; N];

    // The first output frame of RNNoise has a fade-in artifact:
    // feed one silent frame and throw its output away.
    state.process_frame(&mut out_frame, &[0.0f32; N]);

    for chunk in samples.chunks(N) {
        let mut in_frame = [0.0f32; N];
        in_frame[..chunk.len()].copy_from_slice(chunk);
        state.process_frame(&mut out_frame, &in_frame);
        out.extend_from_slice(&out_frame);
    }
    out.truncate(samples.len());
    out
}

async fn ffmpeg(args: &[&str]) -> Result<(), BoxErr> {
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(args)
        .output()
        .await
        .map_err(|e| format!("could not run ffmpeg (is it installed?): {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr)).into())
    }
}

fn path_str(p: &Path) -> Result<&str, BoxErr> {
    p.to_str().ok_or_else(|| "non-utf8 path".into())
}

// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    pretty_env_logger::formatted_builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let token = env::var("BOT_TOKEN").expect("Set the BOT_TOKEN environment variable");
    // Only A-Z a-z 0-9 _ - are allowed by Telegram
    let secret = env::var("WEBHOOK_SECRET").expect("Set the WEBHOOK_SECRET environment variable");

    let state = AppState {
        bot: Bot::new(token),
        webhook_secret: Arc::from(secret),
        webhook_set: Arc::new(AtomicBool::new(false)),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/webhook", post(webhook))
        .with_state(state.clone());

    let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind port");
    println!("Listening on {addr}");

    if let Ok(host) = env::var("RAILWAY_PUBLIC_DOMAIN") {
        let state = state.clone();
        tokio::spawn(async move {
            ensure_webhook(&state, &host).await;
        });
    }

    axum::serve(listener, app).await.expect("Server error");
}
