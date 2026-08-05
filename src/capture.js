// Spawns the WASAPI loopback helper (capture_loopback.py) and turns its raw
// float32 PCM stdout stream into Float32Array chunks for the STT engine.
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Override with RCLI_MEET_PYTHON if `python`/`py` on PATH resolves to the
// Windows Store alias stub instead of a real interpreter.
const PYTHON_EXE = process.env.RCLI_MEET_PYTHON || 'python';
const SCRIPT = path.join(__dirname, '..', 'capture_loopback.py');
const BYTES_PER_SAMPLE = 4; // float32

/**
 * @param onSamples {(samples: Float32Array) => void}
 * @param onFatal {(message: string) => void} called if capture can't run at all
 */
function startCapture(onSamples, onFatal = () => {}) {
  if (!fs.existsSync(SCRIPT)) {
    onFatal(`capture helper not found at ${SCRIPT}`);
    return { stop() {} };
  }

  const proc = spawn(PYTHON_EXE, [SCRIPT], { stdio: ['ignore', 'pipe', 'pipe'] });
  let pending = Buffer.alloc(0);
  let stopped = false;
  // Kept so a non-zero exit can explain *why* (e.g. ModuleNotFoundError:
  // soundcard) instead of just printing an exit code.
  let stderrTail = '';

  // Without this, a missing/unspawnable interpreter emits an unhandled
  // 'error' event, which crashes the whole process with an opaque stack.
  proc.on('error', (err) => {
    if (stopped) return;
    const hint =
      err.code === 'ENOENT'
        ? `\n  "${PYTHON_EXE}" could not be run. Set RCLI_MEET_PYTHON to your real python.exe` +
          `\n  (on Windows, a bare "python" often resolves to the Store alias stub).`
        : '';
    onFatal(`could not start audio capture: ${err.message}${hint}`);
  });

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

    // Copy the remainder instead of keeping a subarray view: a view pins the
    // entire original chunk in memory for the lifetime of the leftover bytes.
    pending =
      usableLen === pending.length ? Buffer.alloc(0) : Buffer.from(pending.subarray(usableLen));
  });

  proc.stderr.on('data', (chunk) => {
    const text = String(chunk);
    stderrTail = (stderrTail + text).slice(-2000);
    // stdout, not stderr -- run.bat redirects stderr to NUL to silence the
    // native addon's log spam, so keep our own diagnostics visible.
    process.stdout.write(`[capture] ${text}`);
  });

  proc.on('exit', (code, signal) => {
    if (stopped) return; // expected teardown
    if (code !== null && code !== 0) {
      const detail = stderrTail.trim() ? `\n${stderrTail.trim()}` : '';
      onFatal(`audio capture stopped unexpectedly (exit code ${code})${detail}`);
    } else if (signal) {
      onFatal(`audio capture stopped unexpectedly (killed by ${signal})`);
    } else {
      onFatal('audio capture stopped unexpectedly (helper exited)');
    }
  });

  return {
    stop() {
      if (stopped) return;
      stopped = true;
      proc.kill();
    },
  };
}

module.exports = { startCapture };
