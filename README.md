# rcli-meet

Offline live-meeting captions + a local LLM you can ask about what was just
said, while it's still being said. No UI, terminal only. Nothing leaves the
machine -- audio never gets uploaded anywhere.

Built on top of [RunAnywhere](https://github.com/RunanywhereAI)'s local
inference engine (Vulkan-accelerated LLM on AMD GPUs) and
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) with Vulkan for
GPU speech-to-text (same ggml stack as the LLM).

## How it works

- **Audio capture**: WASAPI loopback (native to Windows -- no virtual audio
  cable needed) grabs whatever the default output device is currently
  playing. A small Python helper (`capture_audio.py`, using the
  `soundcard` package) streams raw 16kHz mono float32 PCM to the Node
  process over stdout.
- **STT (default: Vulkan Whisper)**: a long-lived GPU worker keeps
  `large-v3-turbo` warm on Vulkan. While someone is speaking, the in-flight
  utterance is re-decoded every ~1.6s and shown live as growing partial
  captions (`… [meeting] words appear here`). After ~0.9s of silence a
  higher-quality beam-search pass commits the final line. A 2–3s lag behind
  live speech is expected and intentional.
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
npm run setup:stt-gpu
```

`setup:stt-gpu` downloads a Vulkan-enabled `whisper-server` (~17 MB) and
`ggml-large-v3-turbo.bin` (~1.6 GB). That model is the accuracy/speed sweet
spot for accented English on an RX 6800XT.

Python 3 with the `soundcard` package is required for the loopback capture
helper:

```
pip install soundcard numpy
```

### STT engines

Selected by `RCLI_MEET_STT_ENGINE`:

| Engine | Default | Device | Notes |
|---|---|---|---|
| `vulkan` | **yes** | GPU (Vulkan / whisper.cpp) | `large-v3-turbo` -- best for Indian English |
| `sensevoice` | | CPU (native sherpa) | FunASR SenseVoice -- strong accents, fast on CPU |
| `whisper` | | CPU (native sherpa) | Whisper-small.en -- weaker than vulkan/turbo |
| `zipformer` | | CPU (native sherpa) | Live word-by-word partials; weak on non-native accents |

**Why Vulkan, not RunAnywhere `loadSTT`?** RunAnywhere's Electron STT uses
sherpa-onnx Whisper with `provider="cpu"` hard-coded -- Vulkan in the SDK
accelerates the LLM only. whisper.cpp with GGML_VULKAN is the matching GPU
path for ASR on AMD.

**Important:** do not use the npm package named `sherpa-onnx` (WASM). This
project uses `sherpa-onnx-node` (native). The WASM package was the previous
root cause of extremely slow, garbled captions.

CPU fallback model downloads:

```
# SenseVoice (recommended CPU fallback for Indian English)
curl -L -o models/sense-voice.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar -xjf models/sense-voice.tar.bz2 -C models/

# Or Whisper-small / Zipformer if you already have them
curl -L -o models/whisper.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2
tar -xjf models/whisper.tar.bz2 -C models/
```

Utterance engines show live growing partials while you speak (Vulkan refreshes
about every 1.6s), then commit a final line after trailing silence.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RCLI_MEET_SDK_DIST` | `D:/the_code/runanywhere/SDK/runanywhere-sdks-main/sdk/runanywhere-electron/dist` | Path to your `@runanywhere/electron` `dist/` build |
| `RCLI_MEET_LLM_PATH` | `qwen2.5-3b` (catalog id, auto-downloads) | Any RunAnywhere LLM catalog id or local GGUF path |
| `RCLI_MEET_EMBEDDER_ID` | `minilm` | RunAnywhere embedder catalog id |
| `RCLI_MEET_PYTHON` | `python` | Python interpreter to run `capture_audio.py`/`play_audio.py` -- override if `python`/`py` on PATH resolves to the Windows Store alias stub instead of a real interpreter |
| `RCLI_MEET_STT_ENGINE` | `vulkan` | `vulkan` \| `sensevoice` \| `whisper` \| `zipformer` |
| `RCLI_MEET_STT_MODEL_DIR` | engine-specific under `models/` | STT model path (file for vulkan, dir for others) |
| `RCLI_MEET_WHISPER_BIN` | `bin/whisper` | Directory containing Vulkan `whisper-server.exe` |
| `RCLI_MEET_WHISPER_MODEL` | `models/ggml-large-v3-turbo.bin` | ggml Whisper model for Vulkan engine |
| `RCLI_MEET_WHISPER_PORT` | `8178` | Local whisper-server port |
| `RCLI_MEET_PARTIAL_MS` | `1600` | How often to refresh live partial captions while speech continues |
| `RCLI_MEET_VAD_THRESHOLD` | `0.012` | Energy level considered "speech" (non-zipformer) |
| `RCLI_MEET_VAD_SILENCE_MS` | `900` | Trailing silence before a final caption is committed |
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

### Commands (typed or spoken, except stop)

| Command | Typed | Spoken | Does |
|---|---|---|---|
| `start` / `record` | Yes | Yes | Begin a new session -- transcribes + stores both sources into `sessions/<date>_<active-window>/` |
| `stop` | Yes | **No** | Stop the current session. Typed-only on purpose: "stop" is an ordinary word, and triggering on every mid-sentence "stop" you say while recording would be worse than requiring it be typed |
| `save` | Yes | Yes | Write the current session's transcript, rolling summary, and metadata to disk |
| `load` | Yes | Yes | Lists saved sessions; type the number to load one into context for Q&A |
| `add <path>` | Yes | (paths aren't speakable) | Ingest a `.txt` file into the current session -- copied into its folder and made searchable immediately |
| anything else | Yes | Yes | A question -- answered from the transcript/summary/files, spoken back unless `--no-tts` |

Spoken commands only trigger when the ENTIRE utterance is just that word (e.g. you said only "start", not "let's start the meeting") -- avoids false triggers during normal conversation.

**Answers are always spoken aloud** (`--no-tts` to disable) via an offline piper voice. While speaking, audio is muted so the system doesn't hear and transcribe its own voice.

**Before `start` / after `stop`**: voice chat mode — only your mic is listened to; meeting loopback is ignored. Speak a question and the answer is spoken back. **After `start`**: mic and meeting audio are both stored as transcript instead.

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

- **LLM** uses RunAnywhere Vulkan. **STT** uses whisper.cpp Vulkan (not
  RunAnywhere `loadSTT`, which is CPU-only today).
- **Use audio with clear speech and no music bed.** A music track in the
  loopback path produces confident nonsense captions, which then poison the
  answers.
- Utterance engines need a trailing silence gap to finalize; give it a beat
  after a pause before asking a question about what was just said.
- Vulkan / SenseVoice / Whisper show "(listening)" mid-utterance; use
  `RCLI_MEET_STT_ENGINE=zipformer` only if you need live word-by-word
  partials and can accept worse accent accuracy.
- Chain-of-thought is suppressed on Qwen models (see above). On other
  reasoning models the `<think>` block is hidden but still consumes budget --
  if you see "used its whole budget reasoning", raise
  `RCLI_MEET_ANSWER_TOKENS`. R1-style distillations reason by training and
  cannot be switched off at the prompt level at all.
- Answer quality is whatever the chosen local LLM gives you -- reliable for
  "what did they say the deadline was", not for deep synthesis.
