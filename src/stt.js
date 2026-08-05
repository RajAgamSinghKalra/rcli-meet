// Streaming speech-to-text over the official sherpa-onnx package (prebuilt,
// real incremental decoding -- not RunAnywhere's own batch-only STT).
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const sherpa_onnx = require('sherpa-onnx');

const SAMPLE_RATE = 16000;

const MODEL_FILES = {
  encoder: 'encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
  decoder: 'decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
  joiner: 'joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
  tokens: 'tokens.txt',
};

const MODEL_URL =
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2';

/**
 * Fail early with an actionable message. Without this, a missing model file
 * surfaces as an opaque native error (or an outright process abort) from
 * inside sherpa-onnx.
 */
function assertModelPresent(modelDir) {
  const missing = Object.values(MODEL_FILES).filter(
    (f) => !fs.existsSync(path.join(modelDir, f))
  );
  if (missing.length === 0) return;

  throw new Error(
    `streaming STT model files missing from:\n  ${modelDir}\n` +
      missing.map((f) => `  - ${f}`).join('\n') +
      `\n\nDownload the model with:\n` +
      `  curl -L -o models/zipformer.tar.bz2 ${MODEL_URL}\n` +
      `  tar -xjf models/zipformer.tar.bz2 -C models/`
  );
}

function createStreamingSTT(modelDir) {
  assertModelPresent(modelDir);

  const recognizer = sherpa_onnx.createOnlineRecognizer({
    modelConfig: {
      transducer: {
        encoder: path.join(modelDir, MODEL_FILES.encoder),
        decoder: path.join(modelDir, MODEL_FILES.decoder),
        joiner: path.join(modelDir, MODEL_FILES.joiner),
      },
      tokens: path.join(modelDir, MODEL_FILES.tokens),
      numThreads: 2,
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

  const stream = recognizer.createStream();
  const emitter = new EventEmitter();
  let lastPartial = '';
  let closed = false;

  emitter.sampleRate = SAMPLE_RATE;

  /** @param samples {Float32Array} mono, 16kHz, range [-1, 1] */
  emitter.feed = function feed(samples) {
    if (closed) return;
    stream.acceptWaveform(SAMPLE_RATE, samples);
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

  // Idempotent: freeing the same native handle twice can crash the process.
  emitter.close = function close() {
    if (closed) return;
    closed = true;
    stream.free();
    recognizer.free();
  };

  return emitter;
}

module.exports = { createStreamingSTT, assertModelPresent, SAMPLE_RATE, MODEL_FILES };
