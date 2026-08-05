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

## Usage

```
npm start -- --minutes 20
```

Captions stream live to the terminal. Type a question and press Enter to get
a streamed answer grounded in the transcript; `/quit` or Ctrl+C exits
cleanly. `--llm <catalog-id-or-path>` and `--embedder <catalog-id>` override
the models per run.

## Honest notes

- Streaming STT is CPU-only (sherpa-onnx has no GPU runtime for this model
  family) -- only the LLM leg is GPU-accelerated.
- Endpoint detection needs a trailing silence gap to finalize an utterance;
  a question asked in the first ~1-1.5s after speech resumes (right after
  the previous endpoint reset) may land before any partial has appeared yet.
- Answer quality is whatever the chosen local LLM gives you -- reliable for
  "what did they say the deadline was", not for deep synthesis.
