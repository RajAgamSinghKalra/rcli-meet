// Spawns the WASAPI loopback helper (capture_loopback.py) and turns its raw
// float32 PCM stdout stream into Float32Array chunks for the STT engine.
const path = require('path');
const { spawn } = require('child_process');

// Override with RCLI_MEET_PYTHON if `python`/`py` on PATH resolves to the
// Windows Store alias stub instead of a real interpreter.
const PYTHON_EXE = process.env.RCLI_MEET_PYTHON || 'python';
const SCRIPT = path.join(__dirname, '..', 'capture_loopback.py');
const BYTES_PER_SAMPLE = 4; // float32

function startCapture(onSamples) {
  const proc = spawn(PYTHON_EXE, [SCRIPT], { stdio: ['ignore', 'pipe', 'pipe'] });
  let pending = Buffer.alloc(0);

  proc.stdout.on('data', (chunk) => {
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;

    const usableLen = pending.length - (pending.length % BYTES_PER_SAMPLE);
    if (usableLen === 0) return;

    const sampleCount = usableLen / BYTES_PER_SAMPLE;
    const floats = new Float32Array(sampleCount);
    // Copy into the Float32Array's own backing buffer (guaranteed aligned)
    // rather than viewing `pending` directly, whose byteOffset may not be a
    // multiple of 4.
    Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength).set(
      pending.subarray(0, usableLen)
    );
    onSamples(floats);

    pending = pending.subarray(usableLen);
  });

  proc.stderr.on('data', (chunk) => {
    // stdout, not stderr -- run.bat redirects stderr to NUL to silence the
    // native addon's log spam, so keep our own diagnostics visible.
    process.stdout.write(`[capture] ${chunk}`);
  });

  proc.on('exit', (code, signal) => {
    if (code !== null && code !== 0) {
      console.log(`[capture] audio capture process exited with code ${code}`);
    } else if (signal) {
      console.log(`[capture] audio capture process killed by ${signal}`);
    }
  });

  return {
    stop() {
      proc.kill();
    },
  };
}

module.exports = { startCapture };
