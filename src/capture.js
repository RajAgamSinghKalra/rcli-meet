// Spawns capture_audio.py (loopback or mic) and turns its raw float32 PCM
// stdout stream into Float32Array chunks for the STT engine.
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Override with RCLI_MEET_PYTHON if `python`/`py` on PATH resolves to the
// Windows Store alias stub instead of a real interpreter.
const PYTHON_EXE = process.env.RCLI_MEET_PYTHON || 'python';
const SCRIPT = path.join(__dirname, '..', 'capture_audio.py');
const BYTES_PER_SAMPLE = 4; // float32

/**
 * @param source {'loopback'|'mic'}
 * @param onSamples {(samples: Float32Array) => void}
 * @param onFatal {(message: string) => void} called if capture can't run at all
 */
function startCapture(source, onSamples, onFatal = () => {}) {
  if (source !== 'loopback' && source !== 'mic') {
    throw new Error(`startCapture: source must be "loopback" or "mic", got "${source}"`);
  }
  if (!fs.existsSync(SCRIPT)) {
    onFatal(`capture helper not found at ${SCRIPT}`);
    return { stop() {} };
  }

  const proc = spawn(PYTHON_EXE, [SCRIPT, '--source', source], { stdio: ['ignore', 'pipe', 'pipe'] });
  let pending = Buffer.alloc(0);
  let stopped = false;
  // Kept so a non-zero exit can explain *why* (e.g. ModuleNotFoundError:
  // soundcard, or "no such device") instead of just printing an exit code.
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
    onFatal(`could not start ${source} capture: ${err.message}${hint}`);
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
    process.stdout.write(`[capture:${source}] ${text}`);
  });

  proc.on('exit', (code, signal) => {
    if (stopped) return; // expected teardown
    if (code !== null && code !== 0) {
      const detail = stderrTail.trim() ? `\n${stderrTail.trim()}` : '';
      onFatal(`${source} capture stopped unexpectedly (exit code ${code})${detail}`);
    } else if (signal) {
      onFatal(`${source} capture stopped unexpectedly (killed by ${signal})`);
    } else {
      onFatal(`${source} capture stopped unexpectedly (helper exited)`);
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
