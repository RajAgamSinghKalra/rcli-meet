// Lightweight energy-based VAD used to segment utterances for offline STT
// (Whisper / SenseVoice / whisper.cpp). Not a model-based VAD -- good enough
// to split speech-vs-silence without an extra model download.
const SAMPLE_RATE = 16000;

const ENERGY_THRESHOLD = Number(process.env.RCLI_MEET_VAD_THRESHOLD) || 0.008;
const MIN_SILENCE_MS = Number(process.env.RCLI_MEET_VAD_SILENCE_MS) || 700;
const MAX_UTTERANCE_MS = Number(process.env.RCLI_MEET_VAD_MAX_MS) || 30000;

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/**
 * @param onSpeechStart {() => void}
 * @param onUtterance {(samples: Float32Array) => void}
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

    flush() {
      if (speaking) finalize();
    },

    /** Drop buffered audio without emitting an utterance. */
    clear() {
      reset();
    },

    get isSpeaking() {
      return speaking;
    },
  };
}

module.exports = {
  createEnergyVad,
  SAMPLE_RATE,
  ENERGY_THRESHOLD,
  MIN_SILENCE_MS,
  MAX_UTTERANCE_MS,
};
