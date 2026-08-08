// Native sherpa-onnx (CPU ONNX Runtime). Never use the WASM `sherpa-onnx`
// package here -- it runs Whisper inside WebAssembly and is an order of
// magnitude slower with worse numeric fidelity than the native addon.
module.exports = require('sherpa-onnx-node');
