# ds4-rs

`ds4-rs` is a **Rust port** of [`ds4.c`](https://github.com/antirez/ds4), a small
native inference engine for DeepSeek V4 Flash. Like the original, it is
intentionally narrow: not a generic GGUF runner, not a wrapper around another
runtime, and not a framework. The main path is a DeepSeek V4 Flash-specific
Metal graph executor with DS4-specific loading, prompt rendering, KV state, and
server API glue.

This Rust port preserves the exact architecture, model constants, tensor layout,
and inference semantics of the C original. The Metal runtime (`ds4_metal.m`) and
all Metal compute shaders (`metal/*.metal`) remain **completely unchanged** —
the Rust code calls them through FFI bindings.

For background, philosophy, speed benchmarks, model weights, and
acknowledgements, please see the [original README](https://github.com/antirez/ds4).

This project is developed with **strong AI assistance**, just like the original.
See the original README for details.

## Layout

```
ds4-rs/
├── Cargo.toml              # Rust project configuration
├── build.rs                # Build script (compiles Metal ObjC runtime + shaders)
├── ds4_metal.m             # ← unchanged: Metal ObjC runtime
├── ds4_metal.h             # ← unchanged: Metal C header
├── metal/*.metal           # ← unchanged: 20 Metal compute kernels
└── src/
    ├── lib.rs              # Library root
    ├── types.rs            # Core type definitions and model constants
    ├── gguf/mod.rs         # GGUF file format parser
    ├── cpu/mod.rs          # CPU reference math kernels
    ├── metal_ffi/mod.rs    # FFI bindings to ds4_metal.m
    ├── engine.rs           # Main engine (model loading, vocabulary, sessions)
    ├── session/mod.rs      # Inference sessions and KV cache
    ├── main.rs             # CLI binary
    └── server_main.rs      # HTTP API server binary
```

### C to Rust Module Map

| C Source | Rust Module | Description |
|---|---|---|
| `ds4.h` | `types.rs` | Public API types, constants, enums |
| `ds4.c` (GGUF parsing) | `gguf/mod.rs` | GGUF v3 model file loading and validation |
| `ds4.c` (CPU math) | `cpu/mod.rs` | F16 conversion, quantization, dot products, RoPE, norms |
| `ds4.c` (engine) | `engine.rs` | Model loading, weight binding, vocabulary, Metal lifecycle |
| `ds4.c` (session/kv) | `session/mod.rs` | KV cache, decode scratch, sampling, checkpoint save/load |
| `ds4_metal.h` | `metal_ffi/mod.rs` | FFI extern declarations and safe wrappers |
| `ds4_metal.m` | *(unchanged)* | Metal device/queue/library, kernel wrappers |
| `metal/*.metal` | *(unchanged)* | 20 Metal compute kernels |
| `ds4_cli.c` | `main.rs` | CLI with one-shot and interactive modes |
| `ds4_server.c` | `server_main.rs` | OpenAI-compatible HTTP API server |

## Building

**Prerequisites:** Rust toolchain (1.75+), macOS 14+ (for Metal backend), Xcode 15+.

### CPU-only build (default)

```sh
cd ds4-rs
cargo build --bin ds4
cargo build --bin ds4-server
```

### Metal backend build

```sh
cd ds4-rs
cargo build --bin ds4 --features metal
cargo build --bin ds4-server --features metal
```

> **Note:** The Metal backend requires macOS 14.0+ (Sonoma) for Metal residency
> set APIs (the original C project has the same requirement). On older macOS
> versions the build falls back to CPU-only mode.

### Release build

```sh
cargo build --release --bin ds4
cargo build --release --bin ds4-server
```

### Run directly

```sh
cargo run --bin ds4 -- --help
cargo run --bin ds4-server -- --help
```

> On first build, the Metal ObjC runtime (`ds4_metal.m`) and all `.metal`
> shader files are compiled automatically by `build.rs`.

## Testing

```sh
cargo test          # 44 unit tests covering all modules
```

Tests cover: GGUF parsing, type conversions, CPU math kernels, argmax/top-k
sampling, KV cache allocation, common-prefix computation, payload serialization,
byte encoding/decoding, top-logprobs, and UTF-8 roundtrips.

## CLI

One-shot prompt:

```sh
cargo run --bin ds4 -- -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
cargo run --bin ds4
ds4>
```

The interactive CLI is a real multi-turn DS4 chat. It keeps the rendered chat
transcript and the live Metal KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

Full CLI options (same as the original C CLI):

```
Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]

Invocation modes:
  ds4
      Start the interactive chat prompt: ds4>
  ds4 -p TEXT
      Run one prompt and exit.
  ds4 --prompt-file FILE
      Run one prompt read from FILE and exit. Useful for long prompts.

Model and runtime:
  -m, --model FILE
      GGUF model path. Default: ds4flash.gguf
  --mtp FILE
      Optional MTP support GGUF used for draft-token probes.
  --mtp-draft N
      Maximum autoregressive MTP draft tokens per speculative step. Default: 1
  --mtp-margin F
      Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3
  -c, --ctx N
      Context size allocated for the session. Default: 32768
  --metal
      Use the Metal graph backend. This is the normal fast path and the default.
  --cpu
      Use the CPU reference/debug backend. Not recommended for normal inference.
  --backend NAME
      Select backend explicitly: metal or cpu. Default: metal
  -t, --threads N
      CPU helper threads for host-side or reference work.
  --quality
      Prefer exact kernels where faster approximate paths exist.
  --warm-weights
      Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.

Prompt and generation:
  -p, --prompt TEXT
      Prompt to generate from.
  --prompt-file FILE
      Read the prompt text from FILE.
  -sys, --system TEXT
      System prompt. Empty string disables the default.
  -n, --tokens N
      Maximum tokens to generate. Default: 50000
  --temp F
      Sampling temperature. 0 is greedy/deterministic. Default: 1
  --top-p F
      Nucleus sampling probability. Default: 1
  --seed N
      Sampling seed for reproducible non-greedy runs.
  --think
      Use normal thinking mode. This is the default.
  --think-max
      Use Think Max when --ctx is at least 393216 tokens.
  --nothink
      Start assistant turns with </think> for direct non-thinking replies.

Interactive commands:
  /help          Show interactive commands.
  /think         Select normal thinking mode.
  /think-max     Select context-gated Think Max mode.
  /nothink       Disable thinking mode.
  /ctx N         Recreate the interactive session with a new context size.
  /read FILE     Read a prompt from FILE and run it.
  /quit, /exit   Leave the interactive prompt.
  Ctrl+C         Stop the current generation and return to the prompt.

Diagnostics:
  --inspect
      Load the model and print a summary only.
  --dump-tokens
      Tokenize -p/--prompt-file exactly as written, then exit.
  --dump-logprobs FILE
      Write greedy continuation top-logprobs as JSON.
  --logprobs-top-k N
      Number of local alternatives stored by --dump-logprobs. Default: 20
  --head-test
      Run the output HC/logits head after the native slice.
  --first-token-test
      Run an exact CPU whole-model pass for the first prompt token.
  --metal-graph-test
      Compare first GPU-resident graph stages with CPU.
  --metal-graph-full-test
      Run the GPU-resident self-token graph across all layers.
  --metal-graph-prompt-test
      Compare CPU and GPU graph logits for the full prompt.

Examples:
  cargo run --bin ds4
  cargo run --bin ds4 -- -p "Write a story about a lazy duck."
  cargo run --bin ds4 -- --think-max --prompt-file prompt.txt --ctx 393216
```

## Server

Start a local OpenAI-compatible server:

```sh
cargo run --release --bin ds4-server -- --ctx 100000
```

The server is Metal-only (requires `--features metal`). It keeps one mutable
graph/KV checkpoint in memory, so stateless clients that resend a longer version
of the same prompt can reuse the shared prefix instead of pre-filling from
token zero.

**Important:** The server requires the Metal backend for inference. Build with
`--features metal` and run on macOS 14+.

### Endpoints

- `GET /v1/models` — List available models
- `POST /v1/chat/completions` — OpenAI-style chat completions
- `POST /v1/completions` — OpenAI-style text completions
- `/health` — Health check

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools`, and `tool_choice`.
Tool schemas are rendered into DeepSeek's DSML tool format, and generated DSML
tool calls are mapped back to OpenAI tool calls.

#### Example

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Server Options

```
ds4-server [options]
  -m, --model FILE         Model path (default: ds4flash.gguf)
  -c, --ctx N              Context size (default: 32768)
  --host ADDR              Bind address (default: 127.0.0.1)
  --port N                 Port (default: 8080)
  --metal / --cpu          Backend
  -t, --threads N          CPU threads
  --quality                Exact kernels
  --disk-cache DIR         Directory for disk KV cache checkpoints
  --max-disk-cache N       Max disk cache entries
  -h, --help               Show this help
```

## KV Cache and Session Checkpoints

The session module implements the full KV cache state machine from the C
original:

- **Raw sliding-window cache** — maintains up to 128 (DS4_N_SWA) KV rows per
  layer in a ring buffer
- **Compressed attention cache** — ratio-4 and ratio-128 layers maintain
  compressed KV rows via learned compressor networks
- **Indexer cache** — ratio-4 layers also maintain an indexer-compressed cache
  for the learned attention indexer
- **Live checkpoint** — the current token prefix, logits, and all per-layer
  cache tensors form a checkpoint that can be extended, rewound, or serialized

### Checkpoint save/load

```rust
// Save checkpoint to disk
let bytes = session.payload_bytes();
session.save_payload(&mut file)?;

// Load checkpoint from disk
session.load_payload(&mut file, bytes)?;
```

The payload format matches the original C format exactly (magic "DSV4",
version 1), so checkpoints are portable between the C and Rust implementations.

## Backends

The default backend is Metal (requires `--features metal`):

```sh
cargo run --bin ds4 --features metal -- -p "Hello" --metal
```

The CPU reference/debug path works without the `metal` feature:

```sh
cargo run --bin ds4 -- -p "Hello" --cpu
```

Do not treat the CPU path as the production target. The server is Metal-only,
and the optimized implementation lives in the Metal graph path.

## License

This project is released under the same license as the original `ds4.c`.
See the `LICENSE` file for details.
