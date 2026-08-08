// Resolve a real Python interpreter on Windows. A bare `python` often hits the
// Microsoft Store alias stub (exit 9009 / "Python was not found") even when a
// working install exists via the `py` launcher.
const { spawnSync } = require('child_process');

const CANDIDATES = [
  process.env.RCLI_MEET_PYTHON,
  process.env.PYTHON,
  'py',
  'python',
  'python3',
].filter(Boolean);

function probe(cmd) {
  try {
    const args = cmd === 'py' ? ['-3', '-c', 'import sys; print(sys.executable)'] : ['-c', 'import sys; print(sys.executable)'];
    const r = spawnSync(cmd, args, { encoding: 'utf8', windowsHide: true, timeout: 8000 });
    if (r.status !== 0) return null;
    const out = (r.stdout || '').trim();
    if (!out || /WindowsApps/i.test(out)) return null;
    return cmd === 'py' ? 'py' : out;
  } catch {
    return null;
  }
}

let cached = undefined;

/** @returns {string} executable name/path; for the py launcher returns "py" */
function resolvePython() {
  if (cached !== undefined) return cached;
  for (const cand of CANDIDATES) {
    const hit = probe(cand);
    if (hit) {
      cached = hit === 'py' ? 'py' : hit;
      return cached;
    }
  }
  cached = process.env.RCLI_MEET_PYTHON || 'python';
  return cached;
}

/** Args prefix so `py` runs as `py -3 script.py ...`. */
function pythonArgs(scriptAndArgs) {
  const exe = resolvePython();
  return exe === 'py' ? ['-3', ...scriptAndArgs] : scriptAndArgs;
}

function spawnPython(scriptAndArgs, opts) {
  const { spawn } = require('child_process');
  const exe = resolvePython();
  return spawn(exe, pythonArgs(scriptAndArgs), opts);
}

module.exports = { resolvePython, pythonArgs, spawnPython };
