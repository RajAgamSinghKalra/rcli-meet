// Offline Whisper speech-to-text with an energy-based VAD segmenter.
//
// This is the default engine (see main.js's RCLI_MEET_STT_ENGINE) because the
// streaming Zipformer model in stt.js is small, English-only, and trained
// mostly on native-accent speech -- it does badly on non-native accents
// (Indian English among them). Whisper was trained on far more diverse,
// heavily-accented multilingual audio and is well documented as meaningfully
// more robust to accents. The trade-off: no true word-by-word streaming --
// a result only arrives once you stop talking (VAD-detected silence), not
// live per-word. The same {createStream} interface as stt.js keeps main.js
// unchanged either way: 'partial' fires once (a "(listening)" placeholder,
// not real text) when speech starts, 'final' fires once decoded.
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const sherpa_onnx = require('sherpa-onnx');

const SAMPLE_RATE = 16000;

const MODEL_FILES = {
  encoder: 'small.en-encoder.int8.onnx',
  decoder: 'small.en-decoder.int8.onnx',
  tokens: 'small.en-tokens.txt',
};

const MODEL_URL = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2';

function assertModelPresent(modelDir) {
  const missing = Object.values(MODEL_FILES).filter((f) => !fs.existsSync(path.join(modelDir, f)));
  if (missing.length === 0) return;
  throw new Error(
    `Whisper STT model files missing from:\n  ${modelDir}\n` +
      missing.map((f) => `  - ${f}`).join('\n') +
      `\n\nDownload with:\n` +
      `  curl -L -o models/whisper.tar.bz2 ${MODEL_URL}\n` +
      `  tar -xjf models/whisper.tar.bz2 -C models/`
  );
}

// Tuned to distinguish speech from room/fan noise on a typical mic/loopback
// setup, not a general-purpose VAD -- override via env if it's too
// trigger-happy or misses quiet speech on your hardware.
const ENERGY_THRESHOLD = Number(process.env.RCLI_MEET_VAD_THRESHOLD) || 0.012;
// How much trailing silence ends an utterance. Shorter = snappier results
// but more risk of cutting a sentence into two decodes; longer = the
// opposite. This is the offline-engine equivalent of the streaming engine's
// rule2MinTrailingSilence.
const MIN_SILENCE_MS = Number(process.env.RCLI_MEET_VAD_SILENCE_MS) || 700;
// Hard cap so continuous talking with no pause can't grow one buffered
// utterance (and its eventual decode latency) without bound.
const MAX_UTTERANCE_MS = 30000;

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/**
 * Buffers audio into speech-bounded utterances by energy. Not a model-based
 * VAD (no extra model download/dependency) -- good enough to segment
 * speech-vs-silence for buffering ahead of an offline decode.
 * @param onSpeechStart {() => void} fires once per utterance, when speech first crosses the threshold
 * @param onUtterance {(samples: Float32Array) => void} fires once per utterance, on trailing silence
 */
function createEnergyVad({ onSpeechStart, onUtterance, sampleRate = SAMPLE_RATE }) {
  let buffer = [];
  let bufferedMs = 0;
  let silenceMs = 0;
  let speaking = false;

  function reset() {
    buffer = [];
    bufferedMs = 0;
    silenceMs = 0;
    speaking = false;
  }

  function finalize() {
    const total = buffer.reduce((n, b) => n + b.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const b of buffer) {
      merged.set(b, offset);
      offset += b.length;
    }
    reset();
    if (merged.length > 0) onUtterance(merged);
  }

  return {
    feed(samples) {
      const durMs = (samples.length / sampleRate) * 1000;
      const loud = rms(samples) > ENERGY_THRESHOLD;

      if (loud) {
        if (!speaking) {
          speaking = true;
          onSpeechStart();
        }
        silenceMs = 0;
        buffer.push(samples);
        bufferedMs += durMs;
      } else if (speaking) {
        // Keep trailing silence in the segment -- a hard cut on the exact
        // threshold crossing clips the tail end of the last word.
        buffer.push(samples);
        bufferedMs += durMs;
        silenceMs += durMs;
        if (silenceMs >= MIN_SILENCE_MS) {
          finalize();
          return;
        }
      }

      if (speaking && bufferedMs >= MAX_UTTERANCE_MS) finalize();
    },

    /** Force-finalize whatever's buffered (e.g. on stream close). */
    flush() {
      if (speaking) finalize();
    },

    get isSpeaking() {
      return speaking;
    },
  };
}

function createSTTEngine(modelDir) {
  assertModelPresent(modelDir);

  const recognizer = sherpa_onnx.createOfflineRecognizer({
    modelConfig: {
      whisper: {
        encoder: path.join(modelDir, MODEL_FILES.encoder),
        decoder: path.join(modelDir, MODEL_FILES.decoder),
        language: 'en',
        task: 'transcribe',
      },
      tokens: path.join(modelDir, MODEL_FILES.tokens),
      numThreads: 2,
      provider: 'cpu',
      debug: 0,
    },
  });

  let engineClosed = false;
  const openStreams = new Set();

  function createStream() {
    if (engineClosed) throw new Error('createStream: STT engine is already closed');

    const emitter = new EventEmitter();
    let streamClosed = false;

    function decode(samples) {
      if (streamClosed || engineClosed) return;
      const offlineStream = recognizer.createStream();
      try {
        offlineStream.acceptWaveform(SAMPLE_RATE, samples);
        recognizer.decode(offlineStream);
        const result = recognizer.getResult(offlineStream);
        const text = (result.text || '').trim();
        if (text) emitter.emit('final', text);
      } finally {
        offlineStream.free();
      }
    }

    const vad = createEnergyVad({
      onSpeechStart: () => emitter.emit('partial', '(listening)'),
      onUtterance: decode,
    });

    emitter.sampleRate = SAMPLE_RATE;
    emitter.feed = function feed(samples) {
      if (streamClosed || engineClosed) return;
      vad.feed(samples);
    };
    emitter.close = function close() {
      if (streamClosed) return;
      streamClosed = true;
      openStreams.delete(emitter);
      vad.flush();
    };

    openStreams.add(emitter);
    return emitter;
  }

  return {
    createStream,
    close() {
      if (engineClosed) return;
      engineClosed = true;
      for (const s of Array.from(openStreams)) s.close();
      recognizer.free();
    },
  };
}

module.exports = { createSTTEngine, assertModelPresent, createEnergyVad, SAMPLE_RATE, MODEL_FILES };
