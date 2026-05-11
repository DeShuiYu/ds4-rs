//! `ds4` — DeepSeek V4 Flash inference CLI.
//!
//! One-shot mode builds a single DeepSeek chat prompt and exits.
//! Interactive mode keeps a token transcript plus one session, so follow-up
//! turns reuse the live Metal KV checkpoint.
//!
//! This mirrors the C CLI in `ds4_cli.c`, keeping policy here and leaving
//! graph/cache mechanics inside the engine API.

use std::io::{self, BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use ds4::engine::Engine;
use ds4::types::*;

// ── Signal handling ────────────────────────────────────────────────────────

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn sigint_handler() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

fn interrupt_requested() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

fn interrupt_clear() {
    INTERRUPTED.store(false, Ordering::SeqCst);
}

// ── CLI configuration ──────────────────────────────────────────────────────

#[derive(Debug)]
struct GenOptions {
    prompt: Option<String>,
    system: String,
    n_predict: i32,
    ctx_size: i32,
    temperature: f32,
    top_p: f32,
    seed: u64,
    dump_tokens: bool,
    dump_logprobs_path: Option<String>,
    dump_logprobs_top_k: i32,
    think_mode: ThinkMode,
    head_test: bool,
    first_token_test: bool,
    metal_graph_test: bool,
    metal_graph_full_test: bool,
    metal_graph_prompt_test: bool,
}

impl Default for GenOptions {
    fn default() -> Self {
        GenOptions {
            prompt: None,
            system: "You are a helpful assistant".to_string(),
            n_predict: 50000,
            ctx_size: 32768,
            temperature: 1.0,
            top_p: 1.0,
            seed: 0,
            dump_tokens: false,
            dump_logprobs_path: None,
            dump_logprobs_top_k: 20,
            think_mode: ThinkMode::High,
            head_test: false,
            first_token_test: false,
            metal_graph_test: false,
            metal_graph_full_test: false,
            metal_graph_prompt_test: false,
        }
    }
}

#[derive(Debug)]
struct CliConfig {
    engine: EngineOptions,
    gen: GenOptions,
    inspect: bool,
}

// ── Help / usage ───────────────────────────────────────────────────────────

fn usage() -> ! {
    eprintln!(
        "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n\
         \n\
         Invocation modes:\n\
           ds4\n\
               Start the interactive chat prompt: ds4>\n\
           ds4 -p TEXT\n\
               Run one prompt and exit.\n\
           ds4 --prompt-file FILE\n\
               Run one prompt read from FILE and exit. Useful for long prompts.\n\
         \n\
         Model and runtime:\n\
           -m, --model FILE\n\
               GGUF model path. Default: ds4flash.gguf\n\
           --mtp FILE\n\
               Optional MTP support GGUF used for draft-token probes.\n\
           --mtp-draft N\n\
               Maximum autoregressive MTP draft tokens per speculative step. Default: 1\n\
           --mtp-margin F\n\
               Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3\n\
           -c, --ctx N\n\
               Context size allocated for the session. Default: 32768\n\
           --metal\n\
               Use the Metal graph backend. This is the normal fast path and the default.\n\
           --cpu\n\
               Use the CPU reference/debug backend. Not recommended for normal inference.\n\
           --backend NAME\n\
               Select backend explicitly: metal or cpu. Default: metal\n\
           -t, --threads N\n\
               CPU helper threads for host-side or reference work.\n\
           --quality\n\
               Prefer exact kernels where faster approximate paths exist.\n\
           --warm-weights\n\
               Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.\n\
         \n\
         Prompt and generation:\n\
           -p, --prompt TEXT\n\
               Prompt to generate from.\n\
           --prompt-file FILE\n\
               Read the prompt text from FILE.\n\
           -sys, --system TEXT\n\
               System prompt. Empty string disables the default. Default: You are a helpful assistant\n\
           -n, --tokens N\n\
               Maximum tokens to generate. Default: 50000\n\
           --temp F\n\
               Sampling temperature. 0 is greedy/deterministic. Default: 1\n\
           --top-p F\n\
               Nucleus sampling probability. Default: 1\n\
           --seed N\n\
               Sampling seed for reproducible non-greedy runs. Default: time-based\n\
           --think\n\
               Use normal thinking mode. This is the default.\n\
           --think-max\n\
               Use Think Max when --ctx is at least 393216 tokens; otherwise normal thinking.\n\
           --nothink\n\
               Start assistant turns with </think> for direct non-thinking replies.\n\
         \n\
         Interactive commands:\n\
           /help          Show interactive commands.\n\
           /think         Select normal thinking mode.\n\
           /think-max     Select context-gated Think Max mode.\n\
           /nothink       Disable thinking mode.\n\
           /ctx N         Recreate the interactive session with a new context size.\n\
           /read FILE     Read a prompt from FILE and run it as the next user message.\n\
           /quit, /exit   Leave the interactive prompt.\n\
           Ctrl+C         Stop the current generation and return to the prompt.\n\
         \n\
         Diagnostics:\n\
           --inspect\n\
               Load the model and print a summary only.\n\
           --dump-tokens\n\
               Tokenize -p/--prompt-file exactly as written, then exit without inference.\n\
           --dump-logprobs FILE\n\
               Write greedy continuation top-logprobs as JSON without printing text.\n\
           --logprobs-top-k N\n\
               Number of local alternatives stored by --dump-logprobs. Default: 20\n\
           --head-test\n\
               Run the output HC/logits head after the native slice.\n\
           --first-token-test\n\
               Run an exact CPU whole-model pass for the first prompt token.\n\
           --metal-graph-test\n\
               Compare first GPU-resident graph stages with CPU.\n\
           --metal-graph-full-test\n\
               Run the GPU-resident self-token graph across all layers.\n\
           --metal-graph-prompt-test\n\
               Compare CPU and GPU graph logits for the full prompt.\n\
         \n\
         Normal CLI commands:\n\
           ./ds4\n\
           ./ds4 -p \"Write a story about a lazy duck.\"\n\
           ./ds4 --think-max --prompt-file prompt.txt --ctx 393216\n\
         \n\
         Notes:\n\
           The CLI keeps KV cache state across interactive turns on the Metal backend.\n\
           Long added input is processed with batched prefill; short continuations use decode.\n\
         \n\
           -h, --help\n\
               Show this help."
    );
    std::process::exit(0);
}

// ── Argument parsing helpers ──────────────────────────────────────────────

fn need_arg(i: &mut usize, args: &[String], opt: &str) -> String {
    if *i + 1 >= args.len() {
        eprintln!("ds4: missing value for {opt}");
        std::process::exit(2);
    }
    *i += 1;
    args[*i].clone()
}

fn parse_int(s: &str, opt: &str) -> i32 {
    let v: i64 = s.parse().unwrap_or_else(|_| {
        eprintln!("ds4: invalid value for {opt}: {s}");
        std::process::exit(2);
    });
    if v <= 0 || v > i32::MAX as i64 {
        eprintln!("ds4: invalid value for {opt}: {s}");
        std::process::exit(2);
    }
    v as i32
}

fn parse_u64(s: &str, opt: &str) -> u64 {
    let v: u64 = s.parse().unwrap_or_else(|_| {
        eprintln!("ds4: invalid value for {opt}: {s}");
        std::process::exit(2);
    });
    if v == 0 {
        eprintln!("ds4: invalid value for {opt}: {s}");
        std::process::exit(2);
    }
    v
}

fn parse_float_range(s: &str, opt: &str, min: f32, max: f32) -> f32 {
    let v: f32 = s.parse().unwrap_or_else(|_| {
        eprintln!("ds4: invalid value for {opt}: {s}");
        std::process::exit(2);
    });
    if !v.is_finite() || v < min || v > max {
        eprintln!("ds4: invalid value for {opt}: {s}");
        std::process::exit(2);
    }
    v
}

// ── Option parsing ────────────────────────────────────────────────────────

fn parse_options(args: &[String]) -> CliConfig {
    let mut cfg = CliConfig {
        engine: EngineOptions::default(),
        gen: GenOptions::default(),
        inspect: false,
    };
    let mut prompt_source: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-p" | "--prompt" => {
                if prompt_source.is_some() {
                    eprintln!("ds4: specify only one prompt source");
                    std::process::exit(2);
                }
                prompt_source = Some(need_arg(&mut i, args, arg));
            }
            "--prompt-file" => {
                if prompt_source.is_some() {
                    eprintln!("ds4: specify only one prompt source");
                    std::process::exit(2);
                }
                let path = need_arg(&mut i, args, arg);
                let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                    eprintln!("ds4: failed to read prompt file '{path}': {e}");
                    std::process::exit(2);
                });
                prompt_source = Some(content);
            }
            "-sys" | "--system" => cfg.gen.system = need_arg(&mut i, args, arg),
            "-m" | "--model" => cfg.engine.model_path = need_arg(&mut i, args, arg),
            "--mtp" => cfg.engine.mtp_path = Some(need_arg(&mut i, args, arg)),
            "--mtp-draft" => {
                cfg.engine.mtp_draft_tokens = parse_int(&need_arg(&mut i, args, arg), arg) as u32
            }
            "--mtp-margin" => {
                cfg.engine.mtp_margin =
                    parse_float_range(&need_arg(&mut i, args, arg), arg, 0.0, 1000.0)
            }
            "-n" | "--tokens" => cfg.gen.n_predict = parse_int(&need_arg(&mut i, args, arg), arg),
            "-c" | "--ctx" => cfg.gen.ctx_size = parse_int(&need_arg(&mut i, args, arg), arg),
            "--temp" => {
                cfg.gen.temperature =
                    parse_float_range(&need_arg(&mut i, args, arg), arg, 0.0, 100.0)
            }
            "--top-p" => {
                cfg.gen.top_p = parse_float_range(&need_arg(&mut i, args, arg), arg, 0.0, 1.0)
            }
            "--seed" => cfg.gen.seed = parse_u64(&need_arg(&mut i, args, arg), arg),
            "--quality" => cfg.engine.quality = true,
            "-t" | "--threads" => {
                cfg.engine.n_threads = parse_int(&need_arg(&mut i, args, arg), arg) as u32
            }
            "--backend" => {
                let name = need_arg(&mut i, args, arg);
                cfg.engine.backend = Backend::from_name(&name);
            }
            "--cpu" => cfg.engine.backend = Backend::Cpu,
            "--metal" => cfg.engine.backend = Backend::Metal,
            "--dump-tokens" => cfg.gen.dump_tokens = true,
            "--dump-logprobs" => cfg.gen.dump_logprobs_path = Some(need_arg(&mut i, args, arg)),
            "--logprobs-top-k" => {
                cfg.gen.dump_logprobs_top_k = parse_int(&need_arg(&mut i, args, arg), arg)
            }
            "--think" => cfg.gen.think_mode = ThinkMode::High,
            "--think-max" => cfg.gen.think_mode = ThinkMode::Max,
            "--nothink" => cfg.gen.think_mode = ThinkMode::None,
            "--head-test" => cfg.gen.head_test = true,
            "--first-token-test" => cfg.gen.first_token_test = true,
            "--metal-graph-test" => {
                cfg.gen.metal_graph_test = true;
                cfg.engine.backend = Backend::Metal;
            }
            "--metal-graph-full-test" => {
                cfg.gen.metal_graph_full_test = true;
                cfg.engine.backend = Backend::Metal;
            }
            "--metal-graph-prompt-test" => {
                cfg.gen.metal_graph_prompt_test = true;
                cfg.engine.backend = Backend::Metal;
            }
            "--inspect" => cfg.inspect = true,
            "--warm-weights" => cfg.engine.warm_weights = true,
            _ => {
                eprintln!("ds4: unknown option: {arg}");
                usage();
            }
        }
        i += 1;
    }

    cfg.gen.prompt = prompt_source;
    cfg
}

// ── Context memory logging ────────────────────────────────────────────────

fn log_context_memory(backend: Backend, ctx_size: i32) {
    let m = Engine::context_memory_estimate(backend, ctx_size);
    log::info!(
        "context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
        m.total_bytes as f64 / (1024.0 * 1024.0),
        ctx_size,
        backend.name(),
        m.prefill_cap,
        m.raw_cap,
        m.comp_cap,
    );
}

// ── Think mode helpers ────────────────────────────────────────────────────

fn effective_think_mode(gen: &GenOptions) -> ThinkMode {
    think_mode_for_context(gen.think_mode, gen.ctx_size)
}

fn think_max_downgraded(gen: &GenOptions) -> bool {
    gen.think_mode == ThinkMode::Max && effective_think_mode(gen) != ThinkMode::Max
}

fn warn_think_max_downgraded(gen: &GenOptions) {
    if !think_max_downgraded(gen) {
        return;
    }
    log::warn!(
        "{} needs --ctx >= {}; ctx={} uses normal thinking instead",
        "--think-max",
        think_max_min_context(),
        gen.ctx_size,
    );
}

// ── Token printer (thinking mode formatting) ──────────────────────────────

struct TokenPrinter {
    format_thinking: bool,
    in_think: bool,
    color_open: bool,
    use_color: bool,
    last_output_newline: bool,
    pending: Vec<u8>,
}

impl TokenPrinter {
    fn new(format_thinking: bool, use_color: bool) -> Self {
        TokenPrinter {
            format_thinking,
            in_think: format_thinking,
            color_open: false,
            use_color,
            last_output_newline: true,
            pending: Vec::with_capacity(16),
        }
    }

    fn set_grey(&mut self) {
        if self.use_color && !self.color_open {
            print!("\x1b[90m");
            self.color_open = true;
        }
    }

    fn reset_color(&mut self) {
        if self.use_color && self.color_open {
            print!("\x1b[0m");
            self.color_open = false;
        }
    }

    fn write_char(&mut self, c: u8) {
        if self.in_think {
            self.set_grey();
        }
        print!("{}", c as char);
        self.last_output_newline = c == b'\n';
    }

    fn process(&mut self, text: &[u8], finish: bool) {
        let think_open = b"<think>";
        let think_close = b"</think>";

        // Build combined buffer from pending + incoming
        let total_len = self.pending.len() + text.len();
        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(&self.pending);
        buf.extend_from_slice(text);
        self.pending.clear();

        let mut i = 0;
        while i < total_len {
            let remaining = &buf[i..];
            let rem = remaining.len();

            if remaining.starts_with(think_open) {
                self.in_think = true;
                i += think_open.len();
                continue;
            }

            if remaining.starts_with(think_close) {
                self.in_think = false;
                self.reset_color();
                if !self.last_output_newline {
                    println!();
                    self.last_output_newline = true;
                }
                i += think_close.len();
                continue;
            }

            if !finish && remaining[0] == b'<' {
                let is_partial_open = think_open.starts_with(remaining) && rem < think_open.len();
                let is_partial_close =
                    think_close.starts_with(remaining) && rem < think_close.len();
                if is_partial_open || is_partial_close {
                    // Save partial as pending
                    self.pending.extend_from_slice(remaining);
                    break;
                }
            }

            self.write_char(remaining[0]);
            i += 1;
        }
    }

    fn finish(&mut self) {
        if self.format_thinking {
            self.process(b"", true);
            self.reset_color();
        }
        io::stdout().flush().ok();
    }

    fn write_text(&mut self, text: &[u8]) {
        if self.format_thinking {
            self.process(text, false);
        } else if !text.is_empty() {
            let s = std::str::from_utf8(text).unwrap_or_default();
            print!("{s}");
            self.last_output_newline = text.last() == Some(&b'\n');
        }
    }
}

// ── Generation done callback ──────────────────────────────────────────────

fn generation_done(printer: &mut TokenPrinter) {
    printer.finish();
    if !printer.last_output_newline {
        println!();
        printer.last_output_newline = true;
    }
    io::stdout().flush().ok();
}

// ── Build prompt ──────────────────────────────────────────────────────────

fn is_rendered_chat_prompt(prompt: &str) -> bool {
    prompt.starts_with("<｜begin▁of▁sentence｜>")
}

fn build_prompt(engine: &Engine, gen: &GenOptions) -> TokenVec {
    let prompt = gen.prompt.as_deref().unwrap_or("");
    let think_mode = effective_think_mode(gen);

    if is_rendered_chat_prompt(prompt) {
        engine.tokenize_rendered(prompt)
    } else {
        let sys = if gen.system.is_empty() {
            None
        } else {
            Some(gen.system.as_str())
        };
        let mut out = TokenVec::new();
        engine.encode_chat_prompt(sys, prompt, think_mode, &mut out);
        out
    }
}

// ── One-shot sampled generation ───────────────────────────────────────────

fn run_sampled_generation(engine: &Engine, cfg: &CliConfig, prompt: &TokenVec) -> Result<()> {
    let _session = engine.create_session(cfg.gen.ctx_size as u32)?;

    let think_mode = effective_think_mode(&cfg.gen);
    let use_color = io::stdout().is_terminal();
    let mut printer = TokenPrinter::new(think_mode.is_enabled(), use_color);

    // Prefill timing
    let t_prefill0 = Instant::now();
    // In the real implementation: session.sync(prompt)
    eprintln!(
        "ds4: prompt has {} tokens (session sync not yet implemented - placeholder)",
        prompt.len()
    );
    let t_prefill1 = Instant::now();

    let _max_tokens = cfg.gen.n_predict;
    // room = ctx - pos, but session isn't real yet
    let generated = 0;
    let t_decode0 = Instant::now();

    // Placeholder generation loop — real session API will go here
    eprintln!(
        "ds4: generation would start here (temperature={}, top-p={})",
        cfg.gen.temperature, cfg.gen.top_p
    );

    let t_decode1 = Instant::now();

    generation_done(&mut printer);

    let prefill_s = t_prefill1.duration_since(t_prefill0).as_secs_f64();
    let decode_s = t_decode1.duration_since(t_decode0).as_secs_f64();
    log::info!(
        "prefill: {:.2} t/s, generation: {:.2} t/s",
        if prefill_s > 0.0 {
            prompt.len() as f64 / prefill_s
        } else {
            0.0
        },
        if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        },
    );

    Ok(())
}

// ── One-shot argmax generation (greedy, temperature=0) ────────────────────

fn run_argmax_generation(_engine: &Engine, cfg: &CliConfig, _prompt: &TokenVec) -> Result<()> {
    // In the C code: ds4_engine_generate_argmax(...)
    // This is a CPU-only/greedy path that doesn't require a Metal session.
    // Placeholder for when the engine API supports this.
    let think_mode = effective_think_mode(&cfg.gen);
    let use_color = io::stdout().is_terminal();
    let mut printer = TokenPrinter::new(think_mode.is_enabled(), use_color);

    eprintln!(
        "ds4: argmax generation (greedy, {} tokens max) - placeholder",
        cfg.gen.n_predict
    );

    generation_done(&mut printer);
    log::info!("generation: 0.00 t/s (placeholder)");
    Ok(())
}

// ── Logprob dump ─────────────────────────────────────────────────────────

fn run_logprob_dump(engine: &Engine, cfg: &CliConfig, _prompt: &TokenVec) -> Result<()> {
    let _session = engine.create_session(cfg.gen.ctx_size as u32)?;

    let path = cfg.gen.dump_logprobs_path.as_deref().unwrap();
    eprintln!("ds4: dump-logprobs to '{path}' (placeholder - session not yet implemented)");

    log::info!("logprobs dump skipped (session unavailable)");
    Ok(())
}

// ── Main generation dispatch ──────────────────────────────────────────────

fn run_generation(engine: &Engine, cfg: &CliConfig) -> Result<()> {
    let prompt = build_prompt(engine, &cfg.gen);

    // Diagnostic tests
    if cfg.gen.head_test {
        eprintln!("ds4: --head-test requested (placeholder)");
    }
    if cfg.gen.first_token_test {
        eprintln!("ds4: --first-token-test requested (placeholder)");
    }
    if cfg.gen.dump_tokens {
        engine.dump_tokens(&prompt);
    }
    if cfg.gen.metal_graph_test {
        eprintln!("ds4: --metal-graph-test requested (placeholder)");
        return Ok(());
    }
    if cfg.gen.metal_graph_full_test {
        eprintln!("ds4: --metal-graph-full-test requested (placeholder)");
        return Ok(());
    }
    if cfg.gen.metal_graph_prompt_test {
        eprintln!("ds4: --metal-graph-prompt-test requested (placeholder)");
        return Ok(());
    }
    if cfg.gen.dump_logprobs_path.is_some() {
        return run_logprob_dump(engine, cfg, &prompt);
    }

    let diagnostic = cfg.gen.dump_tokens || cfg.gen.head_test || cfg.gen.first_token_test;
    if diagnostic {
        if cfg.gen.dump_tokens {
            engine.dump_tokens(&prompt);
        }
        log::info!(
            "diagnostic run completed on the native {} path.",
            engine.backend_name()
        );
        return Ok(());
    }

    if cfg.gen.temperature > 0.0 || engine.has_mtp() {
        run_sampled_generation(engine, cfg, &prompt)
    } else {
        run_argmax_generation(engine, cfg, &prompt)
    }
}

// ── Interactive REPL ──────────────────────────────────────────────────────

fn print_repl_help() {
    println!("Commands:");
    println!("  /help          Show this help.");
    println!("  /think         Use normal thinking mode.");
    println!("  /think-max     Use Think Max only when context is at least 393216 tokens.");
    println!("  /nothink       Disable thinking mode.");
    println!("  /ctx N         Set context size for following prompts.");
    println!("  /read FILE     Read a prompt from FILE and run it.");
    println!("  /quit, /exit   Leave the prompt.");
    println!("  Ctrl+C         Stop generation and return to the prompt.");
}

fn trim(s: &str) -> &str {
    s.trim()
}

fn run_repl(engine: &Engine, cfg: &mut CliConfig) -> Result<()> {
    // Install Ctrl+C handler
    ctrlc::set_handler(sigint_handler).context("failed to install Ctrl+C handler")?;
    interrupt_clear();

    // Initialize the chat transcript with BOS token
    let mut transcript = TokenVec::new();
    engine.chat_begin(&mut transcript);

    // Apply Think Max prefix if needed
    let mut max_prefix_tokens: usize = 0;
    if effective_think_mode(&cfg.gen) == ThinkMode::Max {
        let prefix = think_max_prefix();
        let prefix_tokens = engine.tokenize(prefix);
        // Insert prefix after BOS (position 1)
        let mut new_v = vec![transcript.v[0]];
        new_v.extend_from_slice(&prefix_tokens.v);
        new_v.extend_from_slice(&transcript.v[1..]);
        transcript.v = new_v;
        max_prefix_tokens = prefix_tokens.len();
    }

    // Create session
    let session_result = engine.create_session(cfg.gen.ctx_size as u32);
    let mut session_alive = session_result.is_ok();

    print_repl_help();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        if interrupt_requested() {
            interrupt_clear();
        }

        print!("ds4> ");
        stdout.flush().ok();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                if e.kind() == io::ErrorKind::Interrupted {
                    interrupt_clear();
                    continue;
                }
                bail!("read error: {e}");
            }
        }

        let cmd = trim(&line);
        if cmd.is_empty() {
            continue;
        }

        match cmd {
            "/help" => print_repl_help(),
            "/think" => {
                cfg.gen.think_mode = ThinkMode::High;
                // Remove max prefix if present
                if max_prefix_tokens > 0 {
                    transcript.v.drain(1..1 + max_prefix_tokens);
                    max_prefix_tokens = 0;
                }
                println!("Thinking mode: high.");
            }
            "/think-max" => {
                cfg.gen.think_mode = ThinkMode::Max;
                let active = effective_think_mode(&cfg.gen) == ThinkMode::Max;
                if active && max_prefix_tokens == 0 {
                    let prefix = think_max_prefix();
                    let prefix_tokens = engine.tokenize(prefix);
                    transcript.v.splice(1..1, prefix_tokens.v.iter().cloned());
                    max_prefix_tokens = prefix_tokens.len();
                } else if !active && max_prefix_tokens > 0 {
                    transcript.v.drain(1..1 + max_prefix_tokens);
                    max_prefix_tokens = 0;
                }
                warn_think_max_downgraded(&cfg.gen);
                println!(
                    "Thinking mode: {}.",
                    if active {
                        "max"
                    } else {
                        "high (ctx below 393216)"
                    }
                );
            }
            "/nothink" => {
                cfg.gen.think_mode = ThinkMode::None;
                if max_prefix_tokens > 0 {
                    transcript.v.drain(1..1 + max_prefix_tokens);
                    max_prefix_tokens = 0;
                }
                println!("Thinking mode: none.");
            }
            ctx if ctx.starts_with("/ctx") => {
                let rest = ctx[4..].trim();
                if rest.is_empty() {
                    eprintln!("ds4: /ctx needs a positive integer");
                } else {
                    match rest.parse::<i32>() {
                        Ok(n) if n > 0 => {
                            cfg.gen.ctx_size = n;
                            log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);
                            // Recreate session
                            match engine.create_session(cfg.gen.ctx_size as u32) {
                                Ok(_s) => {
                                    session_alive = true;
                                }
                                Err(e) => {
                                    eprintln!("ds4: failed to recreate session: {e}");
                                    session_alive = false;
                                }
                            }
                            let active =
                                think_mode_for_context(cfg.gen.think_mode, cfg.gen.ctx_size)
                                    == ThinkMode::Max;
                            if active && max_prefix_tokens == 0 {
                                let prefix = think_max_prefix();
                                let prefix_tokens = engine.tokenize(prefix);
                                transcript.v.splice(1..1, prefix_tokens.v.iter().cloned());
                                max_prefix_tokens = prefix_tokens.len();
                            } else if !active && max_prefix_tokens > 0 {
                                transcript.v.drain(1..1 + max_prefix_tokens);
                                max_prefix_tokens = 0;
                            }
                            warn_think_max_downgraded(&cfg.gen);
                        }
                        _ => eprintln!("ds4: /ctx needs a positive integer"),
                    }
                }
            }
            r if r.starts_with("/read") => {
                let path = r[5..].trim();
                if path.is_empty() {
                    eprintln!("ds4: /read needs a file path");
                } else {
                    match std::fs::read_to_string(path) {
                        Ok(content) => {
                            // Run a chat turn
                            eprintln!(
                                "ds4: reading prompt from '{path}' ({}) - placeholder turn",
                                content.len()
                            );
                        }
                        Err(e) => eprintln!("ds4: failed to read file '{path}': {e}"),
                    }
                }
            }
            "/quit" | "/exit" => break,
            _ => {
                if cmd.starts_with('/') {
                    eprintln!("ds4: unknown command: {cmd}");
                    eprintln!("ds4: type /help for commands");
                } else if session_alive {
                    // Run a chat turn with the user's text
                    eprintln!("ds4: interactive generation placeholder for: {cmd}");
                } else {
                    eprintln!("ds4: session unavailable; cannot generate");
                }
            }
        }
    }

    Ok(())
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .format_module_path(false)
        .init();

    let args: Vec<String> = std::env::args().collect();
    // Skip binary name
    let args = if args.is_empty() { &[] } else { &args[1..] };
    let mut cfg = parse_options(args);

    // --dump-tokens special case: just tokenize and exit
    if cfg.gen.dump_tokens {
        let prompt = cfg.gen.prompt.as_deref().unwrap_or("");
        if prompt.is_empty() {
            eprintln!("ds4: --dump-tokens requires -p or --prompt-file");
            return Ok(());
        }
        // Quick standalone tokenization without opening the engine
        // For now, warn and fall through
        eprintln!("ds4: --dump-tokens: standalone tokenization not implemented, opening engine...");
    }

    if !cfg.inspect {
        log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);
        warn_think_max_downgraded(&cfg.gen);
    }

    log::info!(
        "opening model: {} (backend={}, ctx={})",
        cfg.engine.model_path,
        cfg.engine.backend.name(),
        cfg.gen.ctx_size,
    );

    let engine = Engine::open(&cfg.engine)?;

    let result = if cfg.inspect {
        engine.summary();
        Ok(())
    } else if cfg.gen.prompt.is_none() {
        run_repl(&engine, &mut cfg)
    } else {
        run_generation(&engine, &cfg)
    };

    // Drop engine explicitly so we see cleanup logs before exit
    drop(engine);

    result
}
