# rcli-meet

Offline live-meeting captions + a local LLM you can ask about what was just
said, while it's still being said. No UI, terminal only. Nothing leaves the
machine -- audio never gets uploaded anywhere.

Built on top of [RunAnywhere](https://github.com/RunanywhereAI)'s local
inference engine (Vulkan-accelerated on AMD GPUs) and the official
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) streaming Zipformer for
real incremental speech-to-text.

## How it works

- **Audio capture**: WASAPI loopback (native to Windows -- no virtual audio
  cable needed) grabs whatever the default output device is currently
  playing. A small Python helper (`capture_loopback.py`, using the
  `soundcard` package) streams raw 16kHz mono float32 PCM to the Node
  process over stdout.
- **Streaming STT**: the official `sherpa-onnx` npm package runs a streaming
  Zipformer transducer model over that audio incrementally, emitting partial
  captions as they're recognized and finalized segments on speech endpoints.
- **Rolling transcript**: every finalized segment is timestamped, kept in
  memory, and appended to a per-session log file (the ground-truth
  scrollback for every answer).
- **Q&A**: asking a question grounds the LLM in (a) the literal last N
  minutes of transcript, (b) the in-flight (not-yet-finalized) utterance --
  since endpoint detection needs a trailing silence gap, whatever was *just*
  said is often still "partial" -- and (c) a cosine-similarity search over
  every embedded segment for the whole session, so a question can pull in a
  moment from long before the recency window.
- **LLM**: RunAnywhere's engine (llama.cpp under the hood) with Vulkan GPU
  offload -- confirmed working on an AMD RX 6800XT.

## Setup

Requires Windows, Node >= 18, and a RunAnywhere Electron SDK checkout
(`@runanywhere/electron`) with a Vulkan-enabled native addon built for your
platform -- see the [RunAnywhere SDK](https://github.com/RunanywhereAI)
repo.

```
npm install
```

Python 3 with the `soundcard` package is required for the loopback capture
helper:

```
pip install soundcard
```

Download a streaming Zipformer model (English) into `models/`:

```
curl -L -o models/zipformer.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar -xjf models/zipformer.tar.bz2 -C models/
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RCLI_MEET_SDK_DIST` | `D:/the_code/runanywhere/SDK/runanywhere-sdks-main/sdk/runanywhere-electron/dist` | Path to your `@runanywhere/electron` `dist/` build |
| `RCLI_MEET_LLM_PATH` | `qwen2.5-3b` (catalog id, auto-downloads) | Any RunAnywhere LLM catalog id or local GGUF path |
| `RCLI_MEET_EMBEDDER_ID` | `minilm` | RunAnywhere embedder catalog id |
| `RCLI_MEET_PYTHON` | `python` | Python interpreter to run `capture_loopback.py` -- override if `python`/`py` on PATH resolves to the Windows Store alias stub instead of a real interpreter |
| `RCLI_MEET_STT_MODEL_DIR` | `models/sherpa-onnx-streaming-zipformer-en-2023-06-26` | Streaming Zipformer model directory |
| `RCLI_MEET_CONTEXT_TOKENS` | `1200` | Transcript tokens allowed into the prompt (see Context budget) |
| `RCLI_MEET_ANSWER_TOKENS` | `400` | Token budget for the answer, including any reasoning block |
| `RCLI_MEET_THINKING` | unset (off for Qwen) | Set to `on` to re-enable chain-of-thought (see below) |

### Chain-of-thought is suppressed by default on Qwen models

Reasoning models spend most of their token budget inside `<think>` before
answering -- enough that the answer can never arrive. commons exposes a
`disable_thinking` option for this, but the Electron bindings don't surface it,
so rcli-meet appends Qwen3's prompt-level `/no_think` directive (the same
approach the SDK's own `Playground/android-use-agent` uses).

Measured on Qwen3-4B-Q4_K_M with a clean 3-line transcript:

| | Tokens generated | Latency | Answer |
|---|---|---|---|
| Thinking on | 140 | 1558 ms | correct |
| `/no_think` (default) | **12** | **251 ms** | correct |

Same answer, 5.8x faster, and no risk of the budget being eaten before the
answer starts. Set `RCLI_MEET_THINKING=on` to compare. The `<think>` filter
stays active regardless, since R1-style distillations reason by training and
ignore the directive.

### Context budget

The transcript grows without bound; the model's context does not. llama.cpp
computes `available = n_ctx - prompt_tokens - reserved`, silently clamps
`max_tokens` to it, then hard-fails `Prompt too long` once it goes
non-positive. With a typical `n_ctx=2048` fit, feeding an unbounded 20-minute
transcript blows past that within ~10 minutes of speech.

So the prompt is capped at `RCLI_MEET_CONTEXT_TOKENS`, keeping the **most
recent** caption lines plus the in-flight utterance, and it says how many
lines it dropped rather than hiding the truncation. `CONTEXT_TOKENS +
ANSWER_TOKENS` must stay comfortably under your model's `n_ctx` -- if you run
a model with a larger context, raise both.

## Usage

```
node src/quiet.js --minutes 20
```

Or copy `run.example.bat` to `run.bat`, fill in the paths for your machine,
and double-click it each time.

Captions stream live to the terminal. Type a question and press Enter to get
a streamed answer grounded in the transcript; `/quit` or Ctrl+C exits
cleanly. `--llm <catalog-id-or-path>` and `--embedder <catalog-id>` override
the models per run, and `--help` lists everything.

Captions that arrive while an answer is streaming are held back and printed
after it -- repainting the caption line mid-answer emits terminal escape
sequences that shred both.

`src/quiet.js` is the recommended entry point (it's what `run.bat` uses): it
runs `src/main.js` and filters the native addon's `[RAC]` log spam, which
would otherwise bury the captions under ~50 lines per session. The addon logs
at INFO no matter what the host requests -- `rac_init()` stores `cfg.log_level`
in a variable used only by the core's internal logger, while the `RAC_LOG_*`
macros check a separate `min_level` that defaults to INFO and never receives
it, and the bindings don't expose `rac_logger_set_min_level`. Run
`node src/main.js` directly if you want the unfiltered logs for debugging.

## Tests

```
npm test
```

Covers the pure logic with no models or audio required: context budgeting,
reasoning-block filtering, transcript windowing and flush-on-close, retrieval
ranking and its failure handling, and model-path validation.

## Honest notes

- Streaming STT is CPU-only (sherpa-onnx has no GPU runtime for this model
  family) -- only the LLM leg is GPU-accelerated.
- **Use audio with clear speech and no music bed.** A music track in the
  loopback path produces confident nonsense captions, which then poison the
  answers.
- Endpoint detection needs a trailing silence gap to finalize an utterance;
  a question asked in the first ~1-1.5s after speech resumes (right after
  the previous endpoint reset) may land before any partial has appeared yet.
- Chain-of-thought is suppressed on Qwen models (see above). On other
  reasoning models the `<think>` block is hidden but still consumes budget --
  if you see "used its whole budget reasoning", raise
  `RCLI_MEET_ANSWER_TOKENS`. R1-style distillations reason by training and
  cannot be switched off at the prompt level at all.
- Answer quality is whatever the chosen local LLM gives you -- reliable for
  "what did they say the deadline was", not for deep synthesis.
