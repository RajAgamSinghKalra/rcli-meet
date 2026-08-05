// Streaming speech-to-text over the official sherpa-onnx package (WASM, prebuilt,
// real incremental decoding -- not RunAnywhere's own batch-only STT).
const path = require('path');
const { EventEmitter } = require('events');
const sherpa_onnx = require('sherpa-onnx');

const SAMPLE_RATE = 16000;

function createStreamingSTT(modelDir) {
  const recognizer = sherpa_onnx.createOnlineRecognizer({
    modelConfig: {
      transducer: {
        encoder: path.join(modelDir, 'encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx'),
        decoder: path.join(modelDir, 'decoder-epoch-99-avg-1-chunk-16-left-128.onnx'),
        joiner: path.join(modelDir, 'joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx'),
      },
      tokens: path.join(modelDir, 'tokens.txt'),
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

  emitter.sampleRate = SAMPLE_RATE;

  /** @param samples {Float32Array} mono, 16kHz, range [-1, 1] */
  emitter.feed = function feed(samples) {
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

  emitter.close = function close() {
    stream.free();
    recognizer.free();
  };

  return emitter;
}

module.exports = { createStreamingSTT, SAMPLE_RATE };
