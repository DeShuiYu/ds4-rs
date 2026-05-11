//! DS4 HTTP API server — OpenAI/Anthropic-compatible inference server.
//!
//! Architecture: single-threaded accept loop per-connection threads, each
//! connection handler parses exactly one HTTP request, then queues a job to
//! the single Metal worker thread. The worker owns the `ds4::session::Session`
//! and therefore owns all live KV cache state. That keeps session reuse, disk
//! checkpointing, and future batching decisions in one place instead of
//! spreading graph mutations across client threads.
//!
//! Supports:
//! - `POST /v1/chat/completions` — OpenAI-style chat completion
//! - `POST /v1/completions` — OpenAI-style text completion
//! - `POST /v1/models` / `GET /v1/models` — list models
//! - `/health` — health check
//! - Server-Sent Events (SSE) streaming for token-by-token output
//! - Tool/function calling
//! - KV cache checkpoint persistence between requests

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{env, thread};

use anyhow::{bail, Context, Result};
use ds4::engine::Engine;
use ds4::session::Session;
use ds4::types::*;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ── Constants ──────────────────────────────────────────────────────────────

const IO_TIMEOUT_SEC: u64 = 10;
const SEND_STALL_TIMEOUT_MS: u64 = 2000;
const SSE_KEEPALIVE_INTERVAL_MS: u64 = 500;
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_CTX: u32 = 32768;
const DEFAULT_MAX_TOKENS: i32 = 4096;
const DS4_THINK_MAX_MIN_CONTEXT: u32 = 393216;

// ── Signal handling ────────────────────────────────────────────────────────

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

fn install_signal_handlers() -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            // Force exit on second Ctrl+C
            std::process::exit(130);
        }
        STOP_REQUESTED.store(true, Ordering::SeqCst);
        info!("Shutdown requested, waiting for active requests to finish...");
    })
    .context("Failed to install signal handler")?;
    Ok(())
}

// ── Server configuration ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ServerConfig {
    model_path: String,
    ctx_size: u32,
    host: String,
    port: u16,
    backend: Backend,
    n_threads: u32,
    quality: bool,
    disk_cache_dir: Option<String>,
    max_disk_cache_entries: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model_path: "ds4flash.gguf".to_string(),
            ctx_size: DEFAULT_CTX,
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            backend: Backend::Metal,
            n_threads: 0,
            quality: false,
            disk_cache_dir: None,
            max_disk_cache_entries: 100,
        }
    }
}

// ── CLI argument parsing ───────────────────────────────────────────────────

fn parse_args() -> ServerConfig {
    let args: Vec<String> = env::args().collect();
    let mut cfg = ServerConfig::default();

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-m" | "--model" => {
                i += 1;
                if i < args.len() {
                    cfg.model_path = args[i].clone();
                }
            }
            "-c" | "--ctx" => {
                i += 1;
                if i < args.len() {
                    cfg.ctx_size = args[i].parse().unwrap_or(DEFAULT_CTX);
                }
            }
            "--host" => {
                i += 1;
                if i < args.len() {
                    cfg.host = args[i].clone();
                }
            }
            "--port" => {
                i += 1;
                if i < args.len() {
                    cfg.port = args[i].parse().unwrap_or(DEFAULT_PORT);
                }
            }
            "--metal" => {
                cfg.backend = Backend::Metal;
            }
            "--cpu" => {
                cfg.backend = Backend::Cpu;
            }
            "-t" | "--threads" => {
                i += 1;
                if i < args.len() {
                    cfg.n_threads = args[i].parse().unwrap_or(0);
                }
            }
            "--quality" => {
                cfg.quality = true;
            }
            "--disk-cache" => {
                i += 1;
                if i < args.len() {
                    cfg.disk_cache_dir = Some(args[i].clone());
                }
            }
            "--max-disk-cache" => {
                i += 1;
                if i < args.len() {
                    cfg.max_disk_cache_entries = args[i].parse().unwrap_or(100);
                }
            }
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            _ => {
                // skip unknown
            }
        }
        i += 1;
    }

    cfg
}

fn usage() {
    eprintln!(
        "ds4-server [options]
  -m, --model FILE         Model path (default: ds4flash.gguf)
  -c, --ctx N              Context size (default: 32768)
  --host ADDR              Bind address (default: 127.0.0.1)
  --port N                 Port (default: 8080)
  --metal / --cpu          Backend
  -t, --threads N          CPU threads
  --quality                Exact kernels
  --disk-cache DIR         Directory for disk KV cache checkpoints
  --max-disk-cache N       Max disk cache entries
  -h, --help               Show this help"
    );
}

// ── Worker queue ───────────────────────────────────────────────────────────

/// A job submitted by a connection handler to the worker thread.
struct Job {
    /// Original request data
    req: Request,
    /// Channel for the worker to send responses back
    response_tx: Sender<JobResult>,
}

/// Result sent back from the worker to the connection handler.
enum JobResult {
    Streaming(Receiver<StreamEvent>),
    Complete(ResponseData),
    Error(i32, String),
}

/// Events emitted during streaming generation.
enum StreamEvent {
    Token(String),
    ReasoningToken(String),
    ToolCall(ToolCallData),
    Finish {
        content: String,
        reasoning: String,
        tool_calls: Vec<ToolCallData>,
        finish_reason: String,
    },
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCallData {
    id: String,
    name: String,
    arguments: String,
}

/// Parsed request data.
struct Request {
    kind: RequestKind,
    api: ApiStyle,
    model: String,
    messages: Vec<ChatMessage>,
    prompt_text: Option<String>,
    max_tokens: i32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: i32,
    seed: u64,
    stream: bool,
    stream_include_usage: bool,
    stop: Vec<String>,
    tools: Vec<ToolDefinition>,
    tool_choice: ToolChoice,
    think_mode: ThinkMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RequestKind {
    Chat,
    Completion,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ApiStyle {
    OpenAI,
    Anthropic,
}

#[derive(Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
    reasoning_content: Option<String>,
    tool_call_id: Option<String>,
    tool_calls: Vec<ToolCallData>,
}

#[derive(Debug, Clone)]
struct ToolDefinition {
    name: String,
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Clone, PartialEq)]
enum ToolChoice {
    Auto,
    None_,
    Required,
    Function(String),
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice::Auto
    }
}

/// Final response data sent back to client.
struct ResponseData {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCallData>,
    finish_reason: String,
    prompt_tokens: i32,
    completion_tokens: i32,
}

// ── HTTP helpers ───────────────────────────────────────────────────────────

fn http_date() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

fn timestamp_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn set_io_timeout(stream: &TcpStream, secs: u64) -> Result<()> {
    stream
        .set_read_timeout(Some(Duration::from_secs(secs)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(secs)))
        .ok();
    Ok(())
}

fn send_all(mut stream: &TcpStream, data: &[u8]) -> Result<()> {
    let mut written = 0;
    let deadline = std::time::Instant::now() + Duration::from_millis(SEND_STALL_TIMEOUT_MS);

    while written < data.len() {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            bail!("server stopping");
        }
        if std::time::Instant::now() > deadline {
            bail!("send stall timeout");
        }
        match stream.write(&data[written..]) {
            Ok(0) => bail!("connection closed"),
            Ok(n) => {
                written += n;
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::Interrupted =>
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => bail!("send error: {}", e),
        }
    }
    Ok(())
}

// ── HTTP response construction ─────────────────────────────────────────────

fn json_error(code: i32, message: &str) -> String {
    let body = json!({
        "error": {
            "message": message,
            "type": if code == 400 { "invalid_request_error" }
                    else if code == 404 { "not_found_error" }
                    else if code == 503 { "service_unavailable_error" }
                    else { "internal_server_error" },
            "code": code,
        }
    })
    .to_string();
    body
}

fn http_response(code: i32, content_type: &str, body: &str) -> Vec<u8> {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        code,
        reason,
        content_type,
        body.len()
    );
    let mut resp = headers.into_bytes();
    resp.extend_from_slice(body.as_bytes());
    resp
}

fn sse_headers() -> &'static str {
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
}

fn sse_event(event_type: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event_type, data)
}

fn sse_data(data: &str) -> String {
    format!("data: {}\n\n", data)
}

/// Generate a random-ish request ID.
fn make_request_id() -> String {
    use rand::Rng;
    let ts = timestamp_secs();
    let r: u64 = rand::thread_rng().gen();
    format!("chatcmpl-{:x}{:x}", ts, r)
}

// ── Request parsing ────────────────────────────────────────────────────────

fn parse_request_body(method: &str, path: &str, body: &str) -> Result<(Request, bool)> {
    match path {
        "/health" => {
            let req = Request {
                kind: RequestKind::Chat,
                api: ApiStyle::OpenAI,
                model: String::new(),
                messages: vec![],
                prompt_text: None,
                max_tokens: 0,
                temperature: 1.0,
                top_p: 1.0,
                min_p: 0.0,
                top_k: 0,
                seed: 0,
                stream: false,
                stream_include_usage: false,
                stop: vec![],
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                think_mode: ThinkMode::High,
            };
            return Ok((req, false));
        }
        "/v1/models" => {
            let req = Request {
                kind: RequestKind::Chat,
                api: ApiStyle::OpenAI,
                model: String::new(),
                messages: vec![],
                prompt_text: None,
                max_tokens: 0,
                temperature: 1.0,
                top_p: 1.0,
                min_p: 0.0,
                top_k: 0,
                seed: 0,
                stream: false,
                stream_include_usage: false,
                stop: vec![],
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                think_mode: ThinkMode::High,
            };
            return Ok((req, true));
        }
        "/v1/chat/completions" | "/v1/completions" => {}
        _ => bail!("unknown path: {}", path),
    }

    if method != "POST" {
        bail!("method not allowed: {}", method);
    }

    let json: Value = serde_json::from_str(body).context("failed to parse JSON request body")?;

    let kind = if path == "/v1/completions" {
        RequestKind::Completion
    } else {
        RequestKind::Chat
    };

    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("deepseek-v4-flash")
        .to_string();

    let stream = json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let stream_include_usage = json
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let max_tokens = json
        .get("max_tokens")
        .or_else(|| json.get("max_completion_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(DEFAULT_MAX_TOKENS as i64) as i32;

    let temperature = json
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let top_p = json.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let min_p = json.get("min_p").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    let top_k = json.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

    let seed = json
        .get("seed")
        .and_then(|v| v.as_f64())
        .map(|v| if v > 0.0 { v as u64 } else { 0 })
        .unwrap_or(0);

    let stop: Vec<String> = match json.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec![],
    };

    // Tools
    let tools = parse_tools(&json);
    let tool_choice = parse_tool_choice(&json);

    // Thinking / reasoning
    let mut think_mode = ThinkMode::High;
    if let Some(thinking) = json.get("thinking") {
        if let Some(enabled) = thinking.as_bool() {
            if !enabled {
                think_mode = ThinkMode::None;
            }
        } else if thinking.is_object() {
            if let Some(t) = thinking.get("type") {
                if t.as_str() == Some("disabled") {
                    think_mode = ThinkMode::None;
                }
            }
        }
    }
    if let Some(effort) = json.get("reasoning_effort") {
        if let Some(s) = effort.as_str() {
            if s == "max" {
                think_mode = ThinkMode::Max;
            }
        }
    }
    // Model aliases
    if model == "deepseek-chat" {
        think_mode = ThinkMode::None;
    } else if model == "deepseek-reasoner" {
        if think_mode == ThinkMode::None {
            think_mode = ThinkMode::High;
        }
    }

    // Messages
    let messages = if kind == RequestKind::Chat {
        parse_chat_messages(&json)?
    } else {
        vec![]
    };

    let prompt_text = if kind == RequestKind::Completion {
        json.get("prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    let req = Request {
        kind,
        api: ApiStyle::OpenAI,
        model,
        messages,
        prompt_text,
        max_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        seed,
        stream,
        stream_include_usage,
        stop,
        tools,
        tool_choice,
        think_mode,
    };

    Ok((req, false))
}

fn parse_tools(json: &Value) -> Vec<ToolDefinition> {
    let mut tools = vec![];
    if let Some(arr) = json.get("tools").and_then(|v| v.as_array()) {
        for tool_val in arr {
            // OpenAI format: {"type":"function","function":{"name":...,"description":...,"parameters":...}}
            let func = if let Some(f) = tool_val.get("function") {
                f
            } else {
                // Anthropic format: {"name":...,"input_schema":...}
                tool_val
            };

            let name = func
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if name.is_empty() {
                continue;
            }
            let description = func
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let parameters = func
                .get("parameters")
                .or_else(|| func.get("input_schema"))
                .cloned()
                .unwrap_or(json!({}));

            tools.push(ToolDefinition {
                name,
                description,
                parameters,
            });
        }
    }
    tools
}

fn parse_tool_choice(json: &Value) -> ToolChoice {
    match json.get("tool_choice") {
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None_,
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Some(Value::Object(obj)) => {
            if let Some(t) = obj.get("type").and_then(|v| v.as_str()) {
                if t == "function" {
                    if let Some(name) = obj.get("function").and_then(|f| f.as_str()) {
                        return ToolChoice::Function(name.to_string());
                    }
                }
            }
            ToolChoice::Auto
        }
        _ => ToolChoice::Auto,
    }
}

fn parse_chat_messages(json: &Value) -> Result<Vec<ChatMessage>> {
    let mut messages = vec![];
    if let Some(arr) = json.get("messages").and_then(|v| v.as_array()) {
        for msg_val in arr {
            let role = msg_val
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user")
                .to_string();

            let content = extract_content(msg_val);

            let reasoning_content = msg_val
                .get("reasoning_content")
                .and_then(|v| extract_content_value(v));

            let tool_call_id = msg_val
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let tool_calls = parse_message_tool_calls(msg_val);

            messages.push(ChatMessage {
                role,
                content,
                reasoning_content,
                tool_call_id,
                tool_calls,
            });
        }
    }
    Ok(messages)
}

fn extract_content(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            let mut text = String::new();
            for part in arr {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            text
        }
        _ => String::new(),
    }
}

fn extract_content_value(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) => {
            let mut text = String::new();
            for part in arr {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            Some(text)
        }
        _ => None,
    }
}

fn parse_message_tool_calls(msg: &Value) -> Vec<ToolCallData> {
    let mut calls = vec![];
    if let Some(arr) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in arr {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let function = tc.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = function
                .and_then(|f| f.get("arguments"))
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        s.to_string()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| "{}".to_string());
            calls.push(ToolCallData {
                id,
                name,
                arguments,
            });
        }
    }
    calls
}

// ── Prompt rendering ───────────────────────────────────────────────────────

fn render_chat_prompt(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    think_mode: ThinkMode,
) -> String {
    let mut out = String::new();
    out.push_str("<｜begin▁of▁sentence｜>");

    if think_mode == ThinkMode::Max {
        out.push_str(think_max_prefix());
    }

    // Collect system messages
    // Collect system messages into a string directly
    let mut system_text = String::new();
    for msg in messages {
        if msg.role == "system" || msg.role == "developer" {
            if !msg.content.is_empty() {
                if !system_text.is_empty() {
                    system_text.push_str("\n\n");
                }
                system_text.push_str(&msg.content);
            }
        }
    }
    if !tools.is_empty() {
        if !system_text.is_empty() {
            system_text.push_str("\n\n");
        }
        system_text.push_str(&render_tools_prompt(tools));
    }
    if !system_text.is_empty() {
        out.push_str(&system_text);
    }

    // Render conversation
    let think_enabled = think_mode.is_enabled();
    let has_tool_context = !tools.is_empty()
        || messages
            .iter()
            .any(|m| m.role == "assistant" && !m.tool_calls.is_empty());

    let last_user_idx = messages
        .iter()
        .rposition(|m| m.role == "user" || m.role == "tool" || m.role == "function");

    let mut pending_assistant = false;

    for msg in messages {
        match msg.role.as_str() {
            "system" | "developer" => continue,

            "user" => {
                out.push_str("<｜User｜>");
                out.push_str(&msg.content);
                pending_assistant = true;
            }

            "tool" | "function" => {
                out.push_str("<tool_result>");
                out.push_str(&escape_dsml_text(&msg.content));
                out.push_str("</tool_result>");
                pending_assistant = true;
            }

            "assistant" => {
                if pending_assistant {
                    out.push_str("<｜Assistant｜>");
                    if think_enabled {
                        if has_tool_context
                            || messages.iter().position(|m| m.role == "assistant")
                                >= last_user_user_pos(messages, msg)
                        {
                            out.push_str("<think>");
                            if let Some(ref r) = msg.reasoning_content {
                                out.push_str(r);
                            }
                            out.push_str("</think>");
                        } else {
                            out.push_str("</think>");
                        }
                    } else {
                        out.push_str("</think>");
                    }
                }
                out.push_str(&msg.content);
                if !msg.tool_calls.is_empty() {
                    out.push_str(&format_tool_calls_dsml(&msg.tool_calls));
                }
                out.push_str("<｜end▁of▁sentence｜>");
                pending_assistant = false;
            }

            _ => continue,
        }
    }

    if pending_assistant {
        out.push_str("<｜Assistant｜>");
        out.push_str(if think_enabled { "<think>" } else { "</think>" });
    }

    out
}

fn last_user_user_pos(messages: &[ChatMessage], current: &ChatMessage) -> Option<usize> {
    messages
        .iter()
        .position(|m| std::ptr::eq(m, current))
        .or_else(|| {
            messages
                .iter()
                .rposition(|m| m.role == "user" || m.role == "tool" || m.role == "function")
        })
}

fn render_completion_prompt(prompt: &str, think_mode: ThinkMode) -> String {
    let mut out = String::new();
    out.push_str("<｜begin▁of▁sentence｜>");
    if think_mode == ThinkMode::Max {
        out.push_str(think_max_prefix());
    }
    out.push_str("You are a helpful assistant<｜User｜>");
    out.push_str(prompt);
    out.push_str("<｜Assistant｜>");
    out.push_str(if think_mode.is_enabled() {
        "<think>"
    } else {
        "</think>"
    });
    out
}

fn render_tools_prompt(tools: &[ToolDefinition]) -> String {
    let mut out = String::new();
    out.push_str(
        "\n\n## Tools\n\nYou have access to a set of tools to help answer the user question. \
         You can invoke tools by writing a \"\u{ff5c}DSML\u{ff5c}tool_calls>\" block like the following:\n\n\
         \u{ff5c}DSML\u{ff5c}tool_calls>\n\
         \u{ff5c}DSML\u{ff5c}invoke name=\"$TOOL_NAME\">\n\
         \u{ff5c}DSML\u{ff5c}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE\u{ff5c}DSML\u{ff5c}/parameter>\n\
         ...\n\
         \u{ff5c}DSML\u{ff5c}/invoke>\n\
         \u{ff5c}DSML\u{ff5c}/tool_calls>\n\n\
         String parameters should be specified as raw text and set `string=\"true\"`. \
         For all other types (numbers, booleans, arrays, objects), pass the value in JSON format.\n\n",
    );

    for tool in tools {
        out.push_str(&format!(
            "### {}\n\n{}\n\n```json\n{}\n```\n\n",
            tool.name,
            tool.description.as_deref().unwrap_or(""),
            tool.parameters.to_string()
        ));
    }

    out.push_str(
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.",
    );
    out
}

fn escape_dsml_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn format_tool_calls_dsml(calls: &[ToolCallData]) -> String {
    if calls.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\n\n\u{ff5c}DSML\u{ff5c}tool_calls>\n");
    for tc in calls {
        out.push_str(&format!(
            "\u{ff5c}DSML\u{ff5c}invoke name=\"{}\">\n",
            escape_dsml_attr(&tc.name)
        ));
        // Arguments as a JSON parameter
        out.push_str(&format!(
            "\u{ff5c}DSML\u{ff5c}parameter name=\"arguments\" string=\"false\">{}\u{ff5c}DSML\u{ff5c}/parameter>\n",
            tc.arguments
        ));
        out.push_str("\u{ff5c}DSML\u{ff5c}/invoke>\n");
    }
    out.push_str("\u{ff5c}DSML\u{ff5c}/tool_calls>");
    out
}

fn escape_dsml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Tool call parsing from model output ─────────────────────────────────────

/// Parse generated tool calls from DSML in model output.
fn parse_generated_tool_calls(text: &str) -> Vec<ToolCallData> {
    let mut calls = vec![];
    let dsml_calls_start = "\u{ff5c}DSML\u{ff5c}tool_calls>";
    let dsml_calls_end = "\u{ff5c}DSML\u{ff5c}/tool_calls>";
    let dsml_invoke_start = "\u{ff5c}DSML\u{ff5c}invoke";
    let dsml_invoke_end = "\u{ff5c}DSML\u{ff5c}/invoke>";
    let dsml_param_start = "\u{ff5c}DSML\u{ff5c}parameter";
    let dsml_param_end = "\u{ff5c}DSML\u{ff5c}/parameter>";

    // Also support plain XML variants
    let variants: Vec<(&str, &str, &str, &str, &str, &str)> = vec![
        (
            dsml_calls_start,
            dsml_calls_end,
            dsml_invoke_start,
            dsml_invoke_end,
            dsml_param_start,
            dsml_param_end,
        ),
        (
            "<|DSML|tool_calls>",
            "</|DSML|tool_calls>",
            "<|DSML|invoke",
            "</|DSML|invoke>",
            "<|DSML|parameter",
            "</|DSML|parameter>",
        ),
        (
            "<tool_calls>",
            "</tool_calls>",
            "<invoke",
            "</invoke>",
            "<parameter",
            "</parameter>",
        ),
    ];

    for (calls_start, calls_end, invoke_start, invoke_end, param_start, param_end) in &variants {
        if let Some(start_pos) = text.find(calls_start) {
            let rest = &text[start_pos + calls_start.len()..];

            // Find the closing tag
            if let Some(end_pos) = rest.find(calls_end) {
                let body = &rest[..end_pos];

                // Parse invokes
                let mut pos = 0;
                while pos < body.len() {
                    // Skip whitespace
                    while pos < body.len() && body.as_bytes()[pos].is_ascii_whitespace() {
                        pos += 1;
                    }

                    // Check for close
                    if pos + calls_end.len() <= body.len() && body[pos..].starts_with(calls_end) {
                        break;
                    }

                    // Find invoke start
                    if !body[pos..].starts_with(invoke_start) {
                        break;
                    }
                    let tag_start = pos + invoke_start.len();

                    // Find '>' closing the invoke tag
                    let tag_end = body[tag_start..].find('>').map(|i| tag_start + i + 1);
                    let tag_end = match tag_end {
                        Some(e) => e,
                        None => break,
                    };

                    // Extract name from the opening tag
                    let open_tag = &body[pos..tag_end];
                    let name = extract_attr_value(open_tag, "name").unwrap_or_default();

                    pos = tag_end;

                    // Parse parameters until invoke_end
                    let mut arguments = String::new();
                    let mut args: Vec<(String, String, bool)> = vec![];

                    loop {
                        // Skip whitespace
                        while pos < body.len() && body.as_bytes()[pos].is_ascii_whitespace() {
                            pos += 1;
                        }

                        if body[pos..].starts_with(invoke_end) {
                            pos += invoke_end.len();
                            break;
                        }

                        if !body[pos..].starts_with(param_start) {
                            break;
                        }

                        let p_tag_start = pos + param_start.len();
                        let p_tag_end = body[p_tag_start..]
                            .find('>')
                            .map(|i| p_tag_start + i + 1)
                            .unwrap_or(body.len());

                        let p_open_tag = &body[pos..p_tag_end];
                        let p_name = extract_attr_value(p_open_tag, "name").unwrap_or_default();
                        let p_is_string = extract_attr_value(p_open_tag, "string")
                            .map(|s| s == "true")
                            .unwrap_or(true);

                        pos = p_tag_end;

                        // Find param end
                        let value_end = body[pos..].find(param_end).map(|i| pos + i);
                        let value_end = match value_end {
                            Some(e) => e,
                            None => break,
                        };

                        let value = &body[pos..value_end];
                        let decoded = if p_is_string {
                            unescape_dsml_text(value)
                        } else {
                            value.to_string()
                        };

                        args.push((p_name, decoded, p_is_string));
                        pos = value_end + param_end.len();
                    }

                    // Build arguments JSON
                    if !args.is_empty() {
                        let mut arg_json = json!({});
                        if let Some(obj) = arg_json.as_object_mut() {
                            for (name, value, is_string) in &args {
                                if *is_string {
                                    obj.insert(name.clone(), json!(value));
                                } else {
                                    // Try to parse as JSON
                                    match serde_json::from_str::<Value>(value) {
                                        Ok(v) => {
                                            obj.insert(name.clone(), v);
                                        }
                                        Err(_) => {
                                            obj.insert(name.clone(), json!(value));
                                        }
                                    }
                                }
                            }
                        }
                        arguments = arg_json.to_string();
                    } else {
                        arguments = "{}".to_string();
                    }

                    // Generate a tool id
                    let id = format!("call_{:x}", rand::random::<u64>());

                    calls.push(ToolCallData {
                        id,
                        name: name.to_string(),
                        arguments,
                    });
                }

                // Only process the first tool block found
                break;
            }
        }
    }

    calls
}

fn extract_attr_value(tag: &str, attr_name: &str) -> Option<String> {
    let pattern = format!("{}=", attr_name);
    if let Some(start) = tag.find(&pattern) {
        let after_eq = &tag[start + pattern.len()..];
        if let Some(quote_start) = after_eq.find('"') {
            let val_start = quote_start + 1;
            if let Some(quote_end) = after_eq[val_start..].find('"') {
                return Some(after_eq[val_start..val_start + quote_end].to_string());
            }
        }
    }
    None
}

fn unescape_dsml_text(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

// ── Streaming state machine ────────────────────────────────────────────────

enum StreamMode {
    Thinking,
    Text,
    Tool,
    Done,
}

struct StreamState {
    mode: StreamMode,
    emit_pos: usize,
    checked_think_prefix: bool,
    sent_reasoning: bool,
    sent_content: bool,
    tool_calls: Vec<ToolCallData>,
}

impl StreamState {
    fn new(think_enabled: bool) -> Self {
        StreamState {
            mode: if think_enabled {
                StreamMode::Thinking
            } else {
                StreamMode::Text
            },
            emit_pos: 0,
            checked_think_prefix: false,
            sent_reasoning: false,
            sent_content: false,
            tool_calls: vec![],
        }
    }
}

fn process_streaming_text(
    state: &mut StreamState,
    raw: &str,
    has_tools: bool,
    is_final: bool,
) -> Vec<StreamEvent> {
    let mut events = vec![];

    if raw.is_empty() {
        if is_final && matches!(state.mode, StreamMode::Thinking) {
            // Finalize thinking
            state.mode = StreamMode::Text;
        }
        return events;
    }

    let raw_len = raw.len();
    let start = state.emit_pos;
    if start >= raw_len {
        return events;
    }

    match state.mode {
        StreamMode::Thinking => {
            if !state.checked_think_prefix {
                let open = "<think>";
                if raw_len < open.len() && raw.starts_with(open) && !is_final {
                    return events;
                }
                if raw_len >= open.len() && &raw[..open.len()] == open {
                    state.emit_pos = open.len();
                }
                state.checked_think_prefix = true;
            }

            let rest = &raw[state.emit_pos..];
            let close_pos = rest.find("</think>");
            let limit = if let Some(pos) = close_pos {
                state.emit_pos + pos
            } else if is_final {
                raw_len
            } else {
                let hold = "</think>".len() - 1;
                if raw_len > state.emit_pos + hold {
                    let safe = raw_len - hold;
                    safe.min(raw_len)
                } else {
                    state.emit_pos
                }
            };

            if limit > state.emit_pos {
                events.push(StreamEvent::ReasoningToken(
                    raw[state.emit_pos..limit].to_string(),
                ));
                state.sent_reasoning = true;
                state.emit_pos = limit;
            }

            if let Some(pos) = close_pos {
                state.emit_pos = state.emit_pos + pos + "</think>".len();
                state.mode = StreamMode::Text;
            } else if is_final {
                state.mode = StreamMode::Done;
            }
        }

        StreamMode::Text => {
            // Check for tool calls
            let tool_start = if has_tools {
                find_tool_start(&raw[state.emit_pos..])
            } else {
                None
            };

            let limit = if let Some(tpos) = tool_start {
                state.emit_pos + tpos
            } else if is_final {
                raw_len
            } else {
                // Safe limit: don't cut partial tool markers
                let search = &raw[state.emit_pos..];
                let lt_pos = search.rfind('<');
                let limit = if let Some(pos) = lt_pos {
                    let marker = state.emit_pos + pos;
                    // Check if this could be the start of a tool marker
                    let tail = &raw[marker..];
                    if tail.len() < 20 && (tail.starts_with('<') && !tail.contains('>')) {
                        marker
                    } else {
                        raw_len
                    }
                } else {
                    raw_len
                };
                limit
            };

            if limit > state.emit_pos {
                events.push(StreamEvent::Token(raw[state.emit_pos..limit].to_string()));
                state.sent_content = true;
                state.emit_pos = limit;
            }

            if let Some(tpos) = tool_start {
                state.emit_pos = state.emit_pos + tpos;
                // Parse tool calls from the remaining text
                let tool_text = &raw[state.emit_pos..];
                let calls = parse_generated_tool_calls(tool_text);
                for tc in calls {
                    state.tool_calls.push(tc.clone());
                    events.push(StreamEvent::ToolCall(tc));
                }
                state.mode = StreamMode::Done;
            } else if is_final {
                state.mode = StreamMode::Done;
            }
        }

        StreamMode::Tool | StreamMode::Done => {
            // No more content to stream
        }
    }

    events
}

fn find_tool_start(text: &str) -> Option<usize> {
    for marker in &[
        "\u{ff5c}DSML\u{ff5c}tool_calls>",
        "<|DSML|tool_calls>",
        "<tool_calls>",
    ] {
        if let Some(pos) = text.find(marker) {
            return Some(pos);
        }
    }
    None
}

// ── Worker thread ──────────────────────────────────────────────────────────

struct Worker {
    engine: Arc<Engine>,
    session: Mutex<Session>,
    job_rx: Mutex<Receiver<Job>>,
}

impl Worker {
    fn new(engine: Arc<Engine>, ctx_size: u32, job_rx: Receiver<Job>) -> Result<Self> {
        let session = engine
            .create_session(ctx_size)
            .context("Failed to create inference session")?;

        Ok(Worker {
            engine,
            session: Mutex::new(session),
            job_rx: Mutex::new(job_rx),
        })
    }

    fn run(&self) {
        info!("Worker thread started");
        loop {
            let job = match self.job_rx.lock().unwrap().recv() {
                Ok(job) => job,
                Err(_) => {
                    info!("Worker: channel closed, shutting down");
                    break;
                }
            };

            let result = self.process_job(job.req);

            // Send result back
            if let Err(e) = job.response_tx.send(result) {
                error!("Failed to send job result: {}", e);
            }
        }
        info!("Worker thread stopped");
    }

    fn process_job(&self, req: Request) -> JobResult {
        // Render prompt
        let prompt_text = match req.kind {
            RequestKind::Chat => render_chat_prompt(&req.messages, &req.tools, req.think_mode),
            RequestKind::Completion => {
                let text = req.prompt_text.as_deref().unwrap_or("");
                render_completion_prompt(text, req.think_mode)
            }
        };

        // Tokenize
        let tokens = self.engine.tokenize(&prompt_text);
        let prompt_len = tokens.len() as i32;

        if tokens.is_empty() {
            return JobResult::Error(400, "Empty prompt after tokenization".to_string());
        }

        // Sync session
        {
            let mut session = self.session.lock().unwrap();
            if let Err(e) = session.sync(&tokens) {
                return JobResult::Error(500, format!("Session sync failed: {}", e));
            }
        }

        if req.stream {
            // Streaming response
            let (tx, rx) = mpsc::channel();

            // Generate in a separate thread to avoid blocking
            let engine = self.engine.clone();
            let session_ref = &self.session;
            let stop = req.stop.clone();
            let max_tokens = req.max_tokens;
            let temperature = req.temperature;
            let top_p = req.top_p;
            let min_p = req.min_p;
            let top_k = req.top_k;
            let seed = req.seed;
            let has_tools = !req.tools.is_empty() || req.tool_choice != ToolChoice::None_;
            let think_mode = req.think_mode;

            // We need to clone these for the generate thread
            let stop_clone = stop.clone();
            let engine_clone = engine.clone();

            // Spawn generation thread
            let mut session = self.session.lock().unwrap();
            let result_tx = tx.clone();

            // Generate tokens
            let mut generated = String::new();
            let mut completion_tokens = 0;
            let mut finish_reason = "stop".to_string();
            let mut stream_state = StreamState::new(think_mode.is_enabled());

            // Use a random rng for sampling
            let mut local_rng = seed ^ 0x9e3779b97f4a7c15;

            for _ in 0..max_tokens {
                if STOP_REQUESTED.load(Ordering::SeqCst) {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Sample next token
                let token = match session.sample(temperature, top_k, top_p, min_p) {
                    Ok(t) => t,
                    Err(_) => break,
                };

                // Check for EOS
                if token == engine_clone.token_eos() {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Decode token
                let piece = engine_clone
                    .token_text(token)
                    .unwrap_or_else(|| format!("<{}>", token));
                generated.push_str(&piece);
                completion_tokens += 1;

                // Check stop sequences
                let mut stop_found = false;
                for s in &stop_clone {
                    if generated.contains(s) {
                        stop_found = true;
                        // Trim the stop sequence from the output
                        if let Some(pos) = generated.find(s) {
                            generated.truncate(pos);
                        }
                        break;
                    }
                }
                if stop_found {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Eval next position
                if let Err(e) = session.eval(token) {
                    error!("Eval error: {}", e);
                    finish_reason = "error".to_string();
                    break;
                }

                // Check if we've hit context limit
                if session.pos() >= session.ctx() - 1 {
                    finish_reason = "length".to_string();
                    break;
                }
            }

            // Process the generated text through the streaming state machine
            let has_tools = !req.tools.is_empty();
            let events = process_streaming_text(&mut stream_state, &generated, has_tools, true);

            for event in events {
                let _ = result_tx.send(event);
            }

            // Parse final tool calls
            let tool_calls = if !stream_state.tool_calls.is_empty() {
                std::mem::take(&mut stream_state.tool_calls)
            } else {
                parse_generated_tool_calls(&generated)
            };

            // Split reasoning from content
            let (content, reasoning) = split_reasoning_content(&generated);

            let _ = result_tx.send(StreamEvent::Finish {
                content,
                reasoning,
                tool_calls,
                finish_reason,
            });

            drop(result_tx); // Close the channel

            JobResult::Streaming(rx)
        } else {
            // Non-streaming: generate all tokens, collect result
            let mut session = self.session.lock().unwrap();
            let mut generated = String::new();
            let mut completion_tokens = 0;
            let mut finish_reason = "stop".to_string();

            for _ in 0..req.max_tokens {
                if STOP_REQUESTED.load(Ordering::SeqCst) {
                    finish_reason = "stop".to_string();
                    break;
                }

                let token = match session.sample(req.temperature, req.top_k, req.top_p, req.min_p) {
                    Ok(t) => t,
                    Err(_) => break,
                };

                if token == self.engine.token_eos() {
                    finish_reason = "stop".to_string();
                    break;
                }

                let piece = self
                    .engine
                    .token_text(token)
                    .unwrap_or_else(|| format!("<{}>", token));
                generated.push_str(&piece);
                completion_tokens += 1;

                // Check stop sequences
                let mut stop_found = false;
                for s in &req.stop {
                    if generated.contains(s) {
                        stop_found = true;
                        if let Some(pos) = generated.find(s) {
                            generated.truncate(pos);
                        }
                        break;
                    }
                }
                if stop_found {
                    finish_reason = "stop".to_string();
                    break;
                }

                if let Err(e) = session.eval(token) {
                    error!("Eval error: {}", e);
                    finish_reason = "error".to_string();
                    break;
                }

                if session.pos() >= session.ctx() - 1 {
                    finish_reason = "length".to_string();
                    break;
                }
            }

            let (content, reasoning) = split_reasoning_content(&generated);
            let tool_calls = parse_generated_tool_calls(&generated);

            let finish = if !tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                finish_reason
            };

            JobResult::Complete(ResponseData {
                content,
                reasoning,
                tool_calls,
                finish_reason: finish,
                prompt_tokens: prompt_len,
                completion_tokens,
            })
        }
    }
}

fn split_reasoning_content(text: &str) -> (String, String) {
    let open = "<think>";
    let close = "</think>";

    if let Some(start) = text.find(open) {
        let after_open = start + open.len();
        if let Some(end) = text[after_open..].find(close) {
            let reasoning = text[after_open..after_open + end].to_string();
            let content_start = after_open + end + close.len();
            let content = text[content_start..].to_string();
            return (content, reasoning);
        }
        // No closing tag - everything is content (strip opening)
        let content = format!("{}{}", &text[..start], &text[after_open..]);
        return (content, String::new());
    }
    (text.to_string(), String::new())
}

// ── HTTP connection handling ───────────────────────────────────────────────

fn handle_connection(mut stream: TcpStream, job_tx: Sender<Job>, engine_name: String) {
    set_io_timeout(&stream, IO_TIMEOUT_SEC).ok();

    // Read request
    let request = match read_http_request(&mut stream) {
        Ok(req) => req,
        Err(e) => {
            let body = json_error(400, &format!("Bad request: {}", e));
            let resp = http_response(400, "application/json", &body);
            let _ = send_all(&stream, &resp);
            return;
        }
    };

    let path = request.path.clone();
    let method = request.method.clone();
    let body = request.body;

    // Handle health endpoint
    if path == "/health" {
        let status = json!({
            "status": "ok",
            "model": engine_name,
            "timestamp": timestamp_secs(),
        });
        let resp = http_response(200, "application/json", &status.to_string());
        let _ = send_all(&stream, &resp);
        return;
    }

    // Handle models list endpoint
    if path == "/v1/models" {
        let models = json!({
            "object": "list",
            "data": [{
                "id": engine_name,
                "object": "model",
                "created": timestamp_secs(),
                "owned_by": "ds4",
            }]
        });
        let resp = http_response(200, "application/json", &models.to_string());
        let _ = send_all(&stream, &resp);
        return;
    }

    // Parse request body
    let (req, _) = match parse_request_body(&method, &path, &body) {
        Ok(r) => r,
        Err(e) => {
            let body = json_error(400, &format!("Invalid request: {}", e));
            let resp = http_response(400, "application/json", &body);
            let _ = send_all(&stream, &resp);
            return;
        }
    };

    let is_streaming = req.stream;
    let request_id = make_request_id();

    if is_streaming {
        // Send SSE headers
        if let Err(e) = send_all(&stream, sse_headers().as_bytes()) {
            error!("Failed to send SSE headers: {}", e);
            return;
        }

        // Create job for worker
        let (response_tx, response_rx) = mpsc::channel();

        let job = Job { req, response_tx };

        if let Err(e) = job_tx.send(job) {
            error!("Failed to queue job: {}", e);
            let _ = send_all(
                &stream,
                sse_data(&json_error(503, "Server busy").as_str()).as_bytes(),
            );
            return;
        }

        // Process streaming responses
        let created = timestamp_secs();
        let model_name = engine_name.clone();

        // The worker returns JobResult::Streaming with inner Receiver<StreamEvent>
        let event_rx = match response_rx.recv() {
            Ok(JobResult::Streaming(rx)) => rx,
            Ok(JobResult::Error(code, msg)) => {
                let _ = send_all(&stream, sse_data(&json_error(code, &msg)).as_bytes());
                let _ = send_all(&stream, sse_data("[DONE]").as_bytes());
                return;
            }
            _ => {
                let _ = send_all(
                    &stream,
                    sse_data(&json_error(500, "Unexpected response")).as_bytes(),
                );
                let _ = send_all(&stream, sse_data("[DONE]").as_bytes());
                return;
            }
        };

        loop {
            match event_rx.recv() {
                Ok(StreamEvent::Token(text)) => {
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": null,
                        }]
                    });
                    if let Err(e) = send_all(&stream, sse_data(&chunk.to_string()).as_bytes()) {
                        error!("SSE send error: {}", e);
                        break;
                    }
                }
                Ok(StreamEvent::ReasoningToken(text)) => {
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": text},
                            "finish_reason": null,
                        }]
                    });
                    if let Err(e) = send_all(&stream, sse_data(&chunk.to_string()).as_bytes()) {
                        error!("SSE send error: {}", e);
                        break;
                    }
                }
                Ok(StreamEvent::ToolCall(tc)) => {
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    }
                                }]
                            },
                            "finish_reason": null,
                        }]
                    });
                    if let Err(e) = send_all(&stream, sse_data(&chunk.to_string()).as_bytes()) {
                        error!("SSE send error: {}", e);
                        break;
                    }
                }
                Ok(StreamEvent::Finish {
                    content,
                    reasoning,
                    tool_calls,
                    finish_reason,
                }) => {
                    // Send reasoning content if any
                    if !reasoning.is_empty() {
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": reasoning},
                                "finish_reason": null,
                            }]
                        });
                        let _ = send_all(&stream, sse_data(&chunk.to_string()).as_bytes());
                    }
                    // Send content if any
                    if !content.is_empty() {
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": null,
                            }]
                        });
                        let _ = send_all(&stream, sse_data(&chunk.to_string()).as_bytes());
                    }
                    // Send tool calls
                    if !tool_calls.is_empty() {
                        let tool_calls_json: Vec<Value> = tool_calls
                            .iter()
                            .enumerate()
                            .map(|(i, tc)| {
                                json!({
                                    "index": i,
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    }
                                })
                            })
                            .collect();
                        let chunk = json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": tool_calls_json},
                                "finish_reason": null,
                            }]
                        });
                        let _ = send_all(&stream, sse_data(&chunk.to_string()).as_bytes());
                    }
                    // Send final delta with finish_reason
                    let final_chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }]
                    });
                    let _ = send_all(&stream, sse_data(&final_chunk.to_string()).as_bytes());
                    // Send [DONE]
                    let _ = send_all(&stream, sse_data("[DONE]").as_bytes());
                    break;
                }
                Ok(StreamEvent::Error(msg)) => {
                    let _ = send_all(&stream, sse_data(&json_error(500, &msg)).as_bytes());
                    let _ = send_all(&stream, sse_data("[DONE]").as_bytes());
                    break;
                }
                Err(_) => {
                    // Channel closed - generation done
                    break;
                }
            }
        }
    } else {
        // Non-streaming: submit job, wait for result
        let (response_tx, response_rx) = mpsc::channel();

        let job = Job { req, response_tx };

        if let Err(e) = job_tx.send(job) {
            let body = json_error(503, &format!("Server busy: {}", e));
            let resp = http_response(503, "application/json", &body);
            let _ = send_all(&stream, &resp);
            return;
        }

        match response_rx.recv() {
            Ok(JobResult::Complete(data)) => {
                let response = build_openai_response(
                    &request_id,
                    &engine_name,
                    &data.content,
                    &data.reasoning,
                    &data.tool_calls,
                    &data.finish_reason,
                    data.prompt_tokens,
                    data.completion_tokens,
                );
                let resp = http_response(200, "application/json", &response);
                let _ = send_all(&stream, &resp);
            }
            Ok(JobResult::Error(code, msg)) => {
                let body = json_error(code, &msg);
                let resp = http_response(code, "application/json", &body);
                let _ = send_all(&stream, &resp);
            }
            Ok(JobResult::Streaming(_)) => {
                let body = json_error(500, "Unexpected streaming response");
                let resp = http_response(500, "application/json", &body);
                let _ = send_all(&stream, &resp);
            }
            Err(_) => {
                let body = json_error(503, "Worker unavailable");
                let resp = http_response(503, "application/json", &body);
                let _ = send_all(&stream, &resp);
            }
        }
    }
}

// ── HTTP request reading ───────────────────────────────────────────────────

struct HttpRequest {
    method: String,
    path: String,
    _version: String,
    headers: HashMap<String, String>,
    body: String,
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    // We need to read from the original stream without cloning, since cloning
    // a TcpStream creates a separate socket, not the same read position.
    // Use a Vec<u8> buffer to accumulate the request.
    let mut buf = Vec::new();
    let mut temp_buf = [0u8; 65536];
    let mut content_length: usize = 0;
    let mut transfer_chunked = false;

    // Read the request line and headers into buf first
    loop {
        if buf.len() >= 65536 {
            bail!("Request too large");
        }
        let n = stream.read(&mut temp_buf)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&temp_buf[..n]);

        // Check if we have the full headers (double CRLF)
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            // Parse headers to find content-length
            let header_str = String::from_utf8_lossy(&buf[..pos]);
            for line in header_str.lines() {
                if let Some(col) = line.find(':') {
                    let key = line[..col].trim().to_lowercase();
                    let value = line[col + 1..].trim().to_string();
                    if key == "content-length" {
                        content_length = value.parse::<usize>().unwrap_or(0);
                    }
                    if key == "transfer-encoding" && value.to_lowercase().contains("chunked") {
                        transfer_chunked = true;
                    }
                }
            }

            let header_end = pos + 4;
            if transfer_chunked {
                // Read remaining chunked body
                let mut body_buf = buf[header_end..].to_vec();
                loop {
                    // Find a complete chunk
                    let body_str = String::from_utf8_lossy(&body_buf);
                    if let Some(crlf) = body_str.find("\r\n") {
                        let chunk_size_str = body_str[..crlf].trim();
                        if chunk_size_str.is_empty() {
                            // Move past CRLF and continue
                            body_buf.drain(..crlf + 2);
                            continue;
                        }
                        let chunk_size = usize::from_str_radix(chunk_size_str, 16)
                            .context("Invalid chunk size")?;

                        let chunk_start = crlf + 2;
                        if chunk_size == 0 {
                            body_buf.drain(..chunk_start);
                            // Skip trailing CRLF and possible trailers
                            if let Some(final_crlf) =
                                body_buf.windows(4).position(|w| w == b"\r\n\r\n")
                            {
                                body_buf.truncate(final_crlf);
                            } else if body_buf.len() >= 2 {
                                body_buf.drain(..2);
                            }
                            break;
                        }

                        if body_buf.len() >= chunk_start + chunk_size + 2 {
                            // Complete chunk available
                            let chunk_data = &body_buf[chunk_start..chunk_start + chunk_size];
                            body_buf.drain(..chunk_start + chunk_size + 2); // +2 for trailing CRLF
                        } else {
                            // Need to read more data
                            let more = stream.read(&mut temp_buf)?;
                            if more == 0 {
                                break;
                            }
                            body_buf.extend_from_slice(&temp_buf[..more]);
                        }
                    } else {
                        let more = stream.read(&mut temp_buf)?;
                        if more == 0 {
                            break;
                        }
                        body_buf.extend_from_slice(&temp_buf[..more]);
                    }
                }
                buf.truncate(header_end);

                // Now parse the request line and headers properly
                let reader = std::io::BufReader::new(&buf[..]);
                return parse_http_request(reader, &body_buf);
            } else if content_length > 0 {
                // Need to read the body
                let body_end = header_end + content_length;
                while buf.len() < body_end {
                    let n = stream.read(&mut temp_buf)?;
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&temp_buf[..n]);
                }
                let body_buf = buf[header_end..body_end.min(buf.len())].to_vec();
                buf.truncate(header_end);
                let reader = std::io::BufReader::new(&buf[..]);
                return parse_http_request(reader, &body_buf);
            } else {
                // No body
                let reader = std::io::BufReader::new(&buf[..]);
                return parse_http_request(reader, &[]);
            }
        }
    }

    bail!("Failed to parse HTTP request");
}

fn parse_http_request(buf: std::io::BufReader<&[u8]>, body_bytes: &[u8]) -> Result<HttpRequest> {
    let mut lines = buf.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty request"))??;
    let request_line = request_line.trim().to_string();

    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() < 3 {
        bail!("Malformed request line: {}", request_line);
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();
    let version = parts[2].to_string();

    // Read headers from the remaining lines
    let mut headers = HashMap::new();
    for line in &mut lines {
        let line = line?.trim().to_string();
        if line.is_empty() {
            break;
        }
        if let Some(pos) = line.find(':') {
            let key = line[..pos].trim().to_lowercase();
            let value = line[pos + 1..].trim().to_string();
            headers.insert(key, value);
        }
    }

    // Use body_bytes directly (already read by caller)
    let body = String::from_utf8_lossy(body_bytes).to_string();

    Ok(HttpRequest {
        method,
        path,
        _version: version,
        headers,
        body,
    })
}

// ── Response formatting ────────────────────────────────────────────────────

fn build_openai_response(
    id: &str,
    model: &str,
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCallData],
    finish_reason: &str,
    prompt_tokens: i32,
    completion_tokens: i32,
) -> String {
    let mut msg = json!({
        "role": "assistant",
        "content": content,
    });

    if !reasoning.is_empty() {
        msg["reasoning_content"] = json!(reasoning);
    }

    if !tool_calls.is_empty() {
        let tc_array: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                })
            })
            .collect();
        msg["tool_calls"] = json!(tc_array);
    }

    let response = json!({
        "id": id,
        "object": "chat.completion",
        "created": timestamp_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    });

    response.to_string()
}

// ── Main entry point ───────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let config = parse_args();

    info!(
        "DS4 Server starting - model: {}, backend: {}, ctx: {}",
        config.model_path,
        config.backend.name(),
        config.ctx_size
    );

    // Set up signal handling
    install_signal_handlers()?;

    // Open the engine
    let engine_opts = EngineOptions {
        model_path: config.model_path.clone(),
        mtp_path: None,
        backend: config.backend,
        n_threads: config.n_threads,
        mtp_draft_tokens: 1,
        mtp_margin: 3.0,
        warm_weights: false,
        quality: config.quality,
    };

    let engine = Arc::new(Engine::open(&engine_opts).context("Failed to open model engine")?);

    let engine_name = format!("deepseek-v4-flash");

    // Create job channel for worker communication
    let (job_tx, job_rx) = mpsc::channel::<Job>();

    // Start worker thread
    let worker = Arc::new(
        Worker::new(engine.clone(), config.ctx_size, job_rx).context("Failed to create worker")?,
    );

    let worker_handle = {
        let worker = worker.clone();
        thread::Builder::new()
            .name("ds4-worker".into())
            .spawn(move || {
                worker.run();
            })
            .context("Failed to spawn worker thread")?
    };

    // Bind and listen
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).context(format!("Failed to bind to {}", addr))?;

    // Set non-blocking for accept loop
    listener
        .set_nonblocking(true)
        .context("Failed to set non-blocking")?;

    info!("Listening on http://{}", addr);

    // Accept connections loop
    let mut next_id: u64 = 0;

    loop {
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            info!("Shutting down server...");
            break;
        }

        match listener.accept() {
            Ok((stream, peer)) => {
                next_id += 1;
                let conn_id = next_id;

                if STOP_REQUESTED.load(Ordering::SeqCst) {
                    break;
                }

                // Set TCP_NODELAY for low latency streaming
                stream.set_nodelay(true).ok();

                debug!("Connection {} from {}", conn_id, peer);

                let job_tx = job_tx.clone();
                let engine_name = engine_name.clone();

                thread::Builder::new()
                    .name(format!("ds4-conn-{}", conn_id))
                    .spawn(move || {
                        handle_connection(stream, job_tx, engine_name);
                        debug!("Connection {} closed", conn_id);
                    })
                    .ok();
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No pending connection, sleep briefly
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                if !STOP_REQUESTED.load(Ordering::SeqCst) {
                    error!("Accept error: {}", e);
                    std::thread::sleep(Duration::from_millis(100));
                } else {
                    break;
                }
            }
        }
    }

    // Wait for worker to finish
    drop(job_tx);
    worker_handle
        .join()
        .map_err(|_| anyhow::anyhow!("Worker thread panicked"))?;

    info!("Server stopped");
    Ok(())
}

// ── Sampling helpers ───────────────────────────────────────────────────────

fn sample_argmax(logits: &[f32]) -> i32 {
    let mut best = 0i32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i as i32;
        }
    }
    best
}

fn sample_top_p_min_p(
    logits: &[f32],
    temperature: f32,
    _top_k: i32,
    top_p: f32,
    min_p: f32,
    rng_state: &mut u64,
) -> i32 {
    if temperature <= 0.0 {
        return sample_argmax(logits);
    }

    let n = logits.len();
    let mut candidates: Vec<(i32, f32)> = Vec::with_capacity(n);

    // Apply temperature
    let inv_temp = 1.0 / temperature;
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    for (i, &l) in logits.iter().enumerate() {
        let logit = (l - max_logit) * inv_temp;
        if logit < -60.0 {
            continue;
        }
        let prob = logit.exp();
        if prob > 0.0 {
            candidates.push((i as i32, prob));
        }
    }

    if candidates.is_empty() {
        return sample_argmax(logits);
    }

    // Sort by probability descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply min-p: filter out tokens with prob < min_p * max_prob
    let max_prob = candidates[0].1;
    if min_p > 0.0 {
        let threshold = min_p * max_prob;
        candidates.retain(|&(_, p)| p >= threshold);
    }

    if candidates.is_empty() {
        candidates.push((0, 0.0));
    }

    // Apply top-p (nucleus sampling)
    let total: f32 = candidates.iter().map(|&(_, p)| p).sum();
    if total <= 0.0 {
        return candidates[0].0;
    }

    // Normalize
    let threshold = top_p * total;

    let mut cumulative = 0.0f32;
    let mut selected = candidates[0].0;

    for &(idx, prob) in &candidates {
        cumulative += prob;
        selected = idx;
        if cumulative >= threshold {
            break;
        }
    }

    selected
}

fn top_logprobs(logits: &[f32], k: i32) -> Vec<TokenScore> {
    let k = k.max(1) as usize;
    let mut scores: Vec<TokenScore> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let logprob = if l.is_finite() { l } else { f32::NEG_INFINITY };
            TokenScore {
                id: i as i32,
                logit: l,
                logprob,
            }
        })
        .collect();
    scores.sort_by(|a, b| {
        b.logprob
            .partial_cmp(&a.logprob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores.truncate(k);
    scores
}
