// Streaming Zipformer STT via native sherpa-onnx-node.
// Fast word-by-word partials, but weaker on non-native (e.g. Indian) English
// than Vulkan Whisper or SenseVoice -- kept for low-latency demos.
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const sherpa_onnx = require('./sherpa');

const SAMPLE_RATE = 16000;

const MODEL_FILES = {
  encoder: 'encoder-epoch-99-avg-1-chunk-16-left-128.onnx',
  decoder: 'decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
  joiner: 'joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
  tokens: 'tokens.txt',
};

const MODEL_URL =
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2';

function assertModelPresent(modelDir) {
  const missing = Object.values(MODEL_FILES).filter((f) => !fs.existsSync(path.join(modelDir, f)));
  if (missing.length === 0) return;

  throw new Error(
    `streaming STT model files missing from:\n  ${modelDir}\n` +
      missing.map((f) => `  - ${f}`).join('\n') +
      `\n\nDownload the model with:\n` +
      `  curl -L -o models/zipformer.tar.bz2 ${MODEL_URL}\n` +
      `  tar -xjf models/zipformer.tar.bz2 -C models/`
  );
}

function createSTTEngine(modelDir) {
  assertModelPresent(modelDir);

  const recognizer = new sherpa_onnx.OnlineRecognizer({
    modelConfig: {
      transducer: {
        encoder: path.join(modelDir, MODEL_FILES.encoder),
        decoder: path.join(modelDir, MODEL_FILES.decoder),
        joiner: path.join(modelDir, MODEL_FILES.joiner),
      },
      tokens: path.join(modelDir, MODEL_FILES.tokens),
      numThreads: Math.max(2, Math.min(4, require('os').cpus().length - 1 || 2)),
      provider: 'cpu',
      modelType: 'zipformer2',
      debug: 0,
    },
    featConfig: { sampleRate: SAMPLE_RATE, featureDim: 80 },
    decodingMethod: 'greedy_search',
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
  });

  let engineClosed = false;
  const openStreams = new Set();

  function createStream() {
    if (engineClosed) throw new Error('createStream: STT engine is already closed');

    const stream = recognizer.createStream();
    const emitter = new EventEmitter();
    let lastPartial = '';
    let streamClosed = false;

    emitter.sampleRate = SAMPLE_RATE;

    emitter.feed = function feed(samples) {
      if (streamClosed || engineClosed) return;
      stream.acceptWaveform({ samples, sampleRate: SAMPLE_RATE });
      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }

      const result = recognizer.getResult(stream);
      const text = (result.text || '').trim();
      if (text && text !== lastPartial) {
        lastPartial = text;
        emitter.emit('partial', text);
      }

      if (recognizer.isEndpoint(stream)) {
        if (text) emitter.emit('final', text);
        lastPartial = '';
        recognizer.reset(stream);
      }
    };

    emitter.close = function close() {
      if (streamClosed) return;
      streamClosed = true;
      openStreams.delete(emitter);
    };
    emitter.reset = function reset() {
      if (streamClosed || engineClosed) return;
      lastPartial = '';
      recognizer.reset(stream);
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
    },
  };
}

module.exports = { createSTTEngine, assertModelPresent, SAMPLE_RATE, MODEL_FILES };
