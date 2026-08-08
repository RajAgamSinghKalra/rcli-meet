// Offline text-to-speech via native sherpa-onnx-node piper/VITS.
// Playback goes through play_audio.py so it works without a Node audio package.
const path = require('path');
const { spawnPython } = require('./python');
const sherpa_onnx = require('./sherpa');

const VOICE_DIR =
  process.env.RCLI_MEET_TTS_MODEL_DIR ||
  path.join(__dirname, '..', 'models', 'vits-piper-en_US-lessac-medium');

const PLAY_SCRIPT = path.join(__dirname, '..', 'play_audio.py');

function createTTS() {
  const tts = new sherpa_onnx.OfflineTts({
    model: {
      vits: {
        model: path.join(VOICE_DIR, 'en_US-lessac-medium.onnx'),
        tokens: path.join(VOICE_DIR, 'tokens.txt'),
        dataDir: path.join(VOICE_DIR, 'espeak-ng-data'),
      },
    },
    numThreads: 1,
    provider: 'cpu',
  });

  let closed = false;

  return {
    sampleRate: tts.sampleRate,

    synthesize(text) {
      return tts.generate({ text, sid: 0, speed: 1.0 });
    },

    speak(text) {
      if (closed || !text || !text.trim()) return Promise.resolve();
      const { samples, sampleRate } = this.synthesize(text);
      return new Promise((resolve) => {
        const proc = spawnPython([PLAY_SCRIPT, String(sampleRate)], {
          stdio: ['pipe', 'ignore', 'pipe'],
        });
        proc.on('error', () => resolve());
        proc.on('exit', () => resolve());
        proc.stdin.write(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength));
        proc.stdin.end();
      });
    },

    close() {
      if (closed) return;
      closed = true;
    },
  };
}

module.exports = { createTTS };
