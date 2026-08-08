// GPU speech-to-text via whisper.cpp + Vulkan, with LIVE rolling partials.
//
// Whisper is offline/batch, so true word-by-word streaming isn't native -- we
// fake it the standard way: keep buffering speech, re-decode the in-flight
// utterance every ~1.5–2s, and emit the growing transcript as `partial`.
// On trailing silence we do one higher-quality "final" decode and commit.
//
// A long-lived Python ctypes worker keeps large-v3-turbo warm on the GPU.
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const { SAMPLE_RATE, ENERGY_THRESHOLD } = require('./vadEnergy');
const { spawnPython } = require('./python');

const DEFAULT_BIN_DIR = path.join(__dirname, '..', 'bin', 'whisper');
const DEFAULT_MODEL = path.join(__dirname, '..', 'models', 'ggml-large-v3-turbo.bin');
const BIN_DIR = process.env.RCLI_MEET_WHISPER_BIN || DEFAULT_BIN_DIR;
const MODEL_PATH = process.env.RCLI_MEET_WHISPER_MODEL || DEFAULT_MODEL;
const WORKER_SCRIPT = path.join(__dirname, '..', 'whisper_worker.py');
const LANGUAGE = process.env.RCLI_MEET_WHISPER_LANG || 'en';
const INITIAL_PROMPT =
  process.env.RCLI_MEET_WHISPER_PROMPT ||
  'Indian English meeting. Short commands: start, save, load, record, begin. ' +
    'When the user says only "start", transcribe it as start. Transcribe spoken English accurately.';

// Live partial cadence -- user said 2–3s delay is fine.
const PARTIAL_EVERY_MS = Number(process.env.RCLI_MEET_PARTIAL_MS) || 1600;
const MIN_PARTIAL_AUDIO_MS = Number(process.env.RCLI_MEET_MIN_PARTIAL_MS) || 700;
const SILENCE_FINAL_MS = Number(process.env.RCLI_MEET_VAD_SILENCE_MS) || 900;
const MAX_UTTERANCE_MS = Number(process.env.RCLI_MEET_VAD_MAX_MS) || 25000;
const ENERGY_GATE = Number(process.env.RCLI_MEET_VAD_THRESHOLD) || ENERGY_THRESHOLD;

const MODEL_URL =
  'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin';
const BIN_URL =
  'https://github.com/eviscerations/whisper-windows-mcp/releases/download/v1.4.0/whisper-vulkan-win-x64.zip';

function assertModelPresent(modelPath = MODEL_PATH) {
  const missing = [];
  if (!fs.existsSync(path.join(BIN_DIR, 'whisper.dll'))) {
    missing.push(`whisper.dll under ${BIN_DIR}`);
  }
  if (!fs.existsSync(path.join(BIN_DIR, 'ggml-vulkan.dll'))) {
    missing.push(`ggml-vulkan.dll under ${BIN_DIR}`);
  }
  if (!fs.existsSync(modelPath)) missing.push(modelPath);
  if (!fs.existsSync(WORKER_SCRIPT)) missing.push(WORKER_SCRIPT);
  if (missing.length === 0) return;

  throw new Error(
    `Vulkan Whisper STT is not set up yet. Missing:\n` +
      missing.map((m) => `  - ${m}`).join('\n') +
      `\n\nRun once:\n` +
      `  powershell -ExecutionPolicy Bypass -File scripts/setup-stt-gpu.ps1\n` +
      `\nOr download manually:\n` +
      `  binaries: ${BIN_URL}\n` +
      `  model:    ${MODEL_URL}`
  );
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function mergeChunks(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/**
 * Decode queue: partials coalesce to the latest audio (drop stale), finals
 * always run after whatever is in flight so the commit isn't lost.
 */
function createDecodeScheduler(decodeFn) {
  let busy = false;
  let latestPartial = null; // { samples, prompt, resolve, reject }
  let pendingFinal = null;

  async function pump() {
    if (busy) return;
    busy = true;
    try {
      while (latestPartial || pendingFinal) {
        const job = pendingFinal || latestPartial;
        if (job === pendingFinal) pendingFinal = null;
        else latestPartial = null;
        try {
          job.resolve(await decodeFn(job.samples, job.prompt, job.mode));
        } catch (err) {
          job.reject(err);
        }
      }
    } finally {
      busy = false;
      if (latestPartial || pendingFinal) void pump();
    }
  }

  return {
    partial(samples, prompt) {
      return new Promise((resolve, reject) => {
        if (latestPartial) latestPartial.resolve(null); // superseded
        latestPartial = { samples, prompt, mode: 'partial', resolve, reject };
        void pump();
      });
    },
    final(samples, prompt) {
      return new Promise((resolve, reject) => {
        pendingFinal = { samples, prompt, mode: 'final', resolve, reject };
        void pump();
      });
    },
  };
}

/** Strip common Whisper hallucinations on silence / music beds. */
function scrubHallucination(text) {
  const t = String(text || '').trim();
  if (!t) return '';
  const lower = t.toLowerCase();
  const junk = [
    /^thanks? for watching\.?$/i,
    /^thank you\.?$/i,
    /^please subscribe\.?$/i,
    /^subscribe to .+ channel\.?$/i,
    /^clear conversation\.?$/i,
    /^indian english meeting\.?$/i,
    /^transcribe the spoken english accurately\.?$/i,
    /^speakers use indian english accents?\.?$/i,
    /^you$/,
    /^\.+$/,
    /^\[.*\]$/,
    /^\(.*\)$/,
  ];
  if (junk.some((re) => re.test(t))) return '';
  // Tiny fragments that are almost always noise
  if (t.length < 2) return '';
  if (/^(uh+|um+|ah+|hmm+)$/i.test(lower)) return '';
  return t;
}

function createSTTEngine(modelPath = MODEL_PATH) {
  assertModelPresent(modelPath);

  let stderrTail = '';
  let readyResolve;
  let readyReject;
  const ready = new Promise((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });
  let readySettled = false;
  const markReady = () => {
    if (readySettled) return;
    readySettled = true;
    readyResolve();
  };
  const markFailed = (err) => {
    if (readySettled) return;
    readySettled = true;
    readyReject(err);
  };

  const proc = spawnPython(
    [
      WORKER_SCRIPT,
      '--bin-dir',
      BIN_DIR,
      '--model',
      modelPath,
      '--language',
      LANGUAGE,
      '--prompt',
      INITIAL_PROMPT,
    ],
    {
      cwd: path.join(__dirname, '..'),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: { ...process.env },
    }
  );

  proc.on('error', (err) => {
    const hint =
      err.code === 'ENOENT'
        ? `\n  Set RCLI_MEET_PYTHON to your real python.exe if needed.`
        : '';
    markFailed(new Error(`could not start whisper_worker.py: ${err.message}${hint}`));
  });

  proc.stderr.on('data', (chunk) => {
    const text = String(chunk);
    stderrTail = (stderrTail + text).slice(-4000);
    process.stdout.write(`[whisper-gpu] ${text}`);
    if (/\[whisper-worker\] ready/i.test(text)) markReady();
    if (/failed|error/i.test(text) && /whisper_init|failed to get context/i.test(text)) {
      markFailed(new Error(text.trim()));
    }
  });

  proc.on('exit', (code, signal) => {
    if (!readySettled) {
      const detail = stderrTail.trim() ? `\n${stderrTail.trim()}` : '';
      markFailed(
        new Error(`whisper_worker exited before ready (code=${code}, signal=${signal})${detail}`)
      );
    }
  });

  setTimeout(() => {
    if (!readySettled) {
      markFailed(
        new Error(
          `whisper_worker did not become ready in time.\n${stderrTail.trim() || '(no stderr)'}`
        )
      );
    }
  }, 180000).unref?.();

  let stdoutBuf = Buffer.alloc(0);
  const pending = [];

  proc.stdout.on('data', (chunk) => {
    stdoutBuf = Buffer.concat([stdoutBuf, chunk]);
    while (pending.length && stdoutBuf.length >= 4) {
      const n = stdoutBuf.readUInt32LE(0);
      if (stdoutBuf.length < 4 + n) break;
      const text = stdoutBuf.subarray(4, 4 + n).toString('utf8');
      stdoutBuf = stdoutBuf.subarray(4 + n);
      pending.shift().resolve(text.trim());
    }
  });

  /**
   * Protocol v2:
   *   uint32 n_samples | uint8 mode(0=partial,1=final) | uint32 prompt_len | prompt | float32 PCM
   *   n_samples == 0 → shutdown
   */
  function transcribe(samples, prompt, mode) {
    return new Promise((resolve, reject) => {
      if (!proc.stdin.writable) {
        reject(new Error('whisper_worker stdin is closed'));
        return;
      }
      pending.push({ resolve, reject });
      const promptBuf = Buffer.from(String(prompt || ''), 'utf8');
      const header = Buffer.alloc(4 + 1 + 4);
      header.writeUInt32LE(samples.length, 0);
      header.writeUInt8(mode === 'final' ? 1 : 0, 4);
      header.writeUInt32LE(promptBuf.length, 5);
      const pcm = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      try {
        proc.stdin.write(header);
        if (promptBuf.length) proc.stdin.write(promptBuf);
        proc.stdin.write(pcm);
      } catch (err) {
        pending.pop();
        reject(err);
      }
    });
  }

  let engineClosed = false;
  const openStreams = new Set();
  // Shared across meeting + mic streams so GPU isn't double-booked.
  const scheduler = createDecodeScheduler(async (samples, prompt, mode) => {
    await ready;
    let energy = 0;
    for (let i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
    if (samples.length < SAMPLE_RATE * 0.25 || energy / samples.length < 1e-7) return '';
    const text = await transcribe(samples, prompt, mode);
    return scrubHallucination(text);
  });

  function createStream() {
    if (engineClosed) throw new Error('createStream: STT engine is already closed');
    const emitter = new EventEmitter();
    let streamClosed = false;

    const chunks = [];
    let bufferedMs = 0;
    let silenceMs = 0;
    let speaking = false;
    let lastPartialAt = 0;
    let lastPartialText = '';
    let committedPrompt = ''; // prior finals -- conditions Whisper on meeting context
    let decodeGen = 0; // ignore stale async results after reset

    function resetUtterance() {
      chunks.length = 0;
      bufferedMs = 0;
      silenceMs = 0;
      speaking = false;
      lastPartialAt = 0;
      lastPartialText = '';
      decodeGen++;
    }

    function contextPrompt() {
      const tail = committedPrompt.slice(-240);
      return [INITIAL_PROMPT, tail].filter(Boolean).join(' ');
    }

    function snapshot() {
      return mergeChunks(chunks);
    }

    function requestPartial() {
      if (streamClosed || engineClosed || !speaking) return;
      if (bufferedMs < MIN_PARTIAL_AUDIO_MS) return;
      const gen = decodeGen;
      const samples = snapshot();
      const started = Date.now();
      lastPartialAt = started;
      scheduler
        .partial(samples, contextPrompt())
        .then((text) => {
          if (streamClosed || engineClosed || gen !== decodeGen) return;
          if (text == null) return; // superseded
          if (!text || text === lastPartialText) return;
          lastPartialText = text;
          emitter.emit('partial', text);
        })
        .catch((err) => {
          if (!streamClosed && !engineClosed) emitter.emit('error', err);
        });
    }

    function requestFinal() {
      if (streamClosed || engineClosed) return;
      if (bufferedMs < 200) {
        resetUtterance();
        return;
      }
      const samples = snapshot();
      const prompt = contextPrompt();
      // Bump generation so any in-flight partial result is ignored.
      decodeGen++;
      chunks.length = 0;
      bufferedMs = 0;
      silenceMs = 0;
      speaking = false;
      lastPartialAt = 0;
      lastPartialText = '';

      scheduler
        .final(samples, prompt)
        .then((text) => {
          if (streamClosed || engineClosed) return;
          lastPartialText = '';
          const t = text || '';
          if (t) {
            committedPrompt = `${committedPrompt} ${t}`.trim().slice(-500);
            emitter.emit('final', t);
          }
        })
        .catch((err) => {
          if (!streamClosed && !engineClosed) emitter.emit('error', err);
        });
    }

    emitter.sampleRate = SAMPLE_RATE;
    emitter.feed = function feed(samples) {
      if (streamClosed || engineClosed) return;
      const durMs = (samples.length / SAMPLE_RATE) * 1000;
      const loud = rms(samples) > ENERGY_GATE;

      if (loud) {
        if (!speaking) {
          speaking = true;
          silenceMs = 0;
          lastPartialAt = Date.now();
          emitter.emit('partial', '…');
        }
        silenceMs = 0;
        // capture.js already hands us a fresh Float32Array -- keep the reference.
        chunks.push(samples);
        bufferedMs += durMs;

        const now = Date.now();
        if (now - lastPartialAt >= PARTIAL_EVERY_MS) {
          requestPartial();
        }
      } else if (speaking) {
        chunks.push(samples);
        bufferedMs += durMs;
        silenceMs += durMs;
        // Keep refreshing partials during short pauses mid-sentence.
        const now = Date.now();
        if (now - lastPartialAt >= PARTIAL_EVERY_MS && silenceMs < SILENCE_FINAL_MS) {
          requestPartial();
        }
        if (silenceMs >= SILENCE_FINAL_MS) {
          requestFinal();
          return;
        }
      }

      if (speaking && bufferedMs >= MAX_UTTERANCE_MS) {
        requestFinal();
      }
    };

    emitter.close = function close() {
      if (streamClosed) return;
      streamClosed = true;
      openStreams.delete(emitter);
      if (speaking && bufferedMs > 200) requestFinal();
      else resetUtterance();
    };

    /** Drop in-flight audio without emitting a final (used when leaving record mode). */
    emitter.reset = function reset() {
      resetUtterance();
    };

    openStreams.add(emitter);
    return emitter;
  }

  return {
    createStream,
    ready: () => ready,
    close() {
      if (engineClosed) return;
      engineClosed = true;
      for (const s of Array.from(openStreams)) s.close();
      try {
        const header = Buffer.alloc(9);
        header.writeUInt32LE(0, 0);
        if (proc.stdin.writable) proc.stdin.end(header);
      } catch {
        /* ignore */
      }
      try {
        proc.kill();
      } catch {
        /* ignore */
      }
    },
  };
}

module.exports = {
  createSTTEngine,
  assertModelPresent,
  scrubHallucination,
  SAMPLE_RATE,
  MODEL_PATH,
  BIN_DIR,
  PARTIAL_EVERY_MS,
};
