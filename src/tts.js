// Offline text-to-speech via sherpa-onnx's VITS/piper voice (same package
// already used for streaming STT, just a different model). Playback goes
// through play_audio.py so it works without any Node audio-output package.
const path = require('path');
const { spawn } = require('child_process');
const sherpa_onnx = require('sherpa-onnx');

const VOICE_DIR =
  process.env.RCLI_MEET_TTS_MODEL_DIR ||
  path.join(__dirname, '..', 'models', 'vits-piper-en_US-lessac-medium');

const PYTHON_EXE = process.env.RCLI_MEET_PYTHON || 'python';
const PLAY_SCRIPT = path.join(__dirname, '..', 'play_audio.py');

function createTTS() {
  const tts = sherpa_onnx.createOfflineTts({
    offlineTtsModelConfig: {
      offlineTtsVitsModelConfig: {
        model: path.join(VOICE_DIR, 'en_US-lessac-medium.onnx'),
        tokens: path.join(VOICE_DIR, 'tokens.txt'),
        dataDir: path.join(VOICE_DIR, 'espeak-ng-data'),
      },
      numThreads: 1,
      debug: 0,
      provider: 'cpu',
    },
  });

  let closed = false;

  return {
    sampleRate: tts.sampleRate,

    /** @returns {{samples: Float32Array, sampleRate: number}} */
    synthesize(text) {
      return tts.generate({ text, sid: 0, speed: 1.0 });
    },

    /**
     * Synthesize and play through the default speaker. Resolves once
     * playback finishes (so callers can avoid feeding the meeting-audio STT
     * stream while the system's own voice is playing -- otherwise it hears
     * and transcribes itself).
     */
    speak(text) {
      if (closed || !text || !text.trim()) return Promise.resolve();
      const { samples, sampleRate } = this.synthesize(text);
      return new Promise((resolve) => {
        const proc = spawn(PYTHON_EXE, [PLAY_SCRIPT, String(sampleRate)], { stdio: ['pipe', 'ignore', 'pipe'] });
        proc.on('error', () => resolve()); // best-effort: a broken TTS playback shouldn't crash the session
        proc.on('exit', () => resolve());
        proc.stdin.write(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength));
        proc.stdin.end();
      });
    },

    close() {
      if (closed) return;
      closed = true;
      tts.free();
    },
  };
}

module.exports = { createTTS };
