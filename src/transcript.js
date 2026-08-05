// Rolling transcript: timestamped segments in memory, appended live to a
// per-session log file (the scrollback "ground truth" for every LLM answer).
const fs = require('fs');
const path = require('path');

function fmtElapsed(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = String(Math.floor(totalSec / 3600)).padStart(2, '0');
  const m = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
  const s = String(totalSec % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function createTranscript(sessionDir, { onError = () => {} } = {}) {
  fs.mkdirSync(sessionDir, { recursive: true });
  const startedAt = Date.now();
  const logPath = path.join(sessionDir, `session-${startedAt}.log`);
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });
  // Without a listener, a write failure (disk full, permissions) emits an
  // unhandled 'error' event and takes the whole process down mid-session.
  logStream.on('error', (err) => onError(`session log write failed: ${err.message}`));

  const segments = [];
  let closed = false;

  return {
    logPath,

    /** Append a finalized caption segment; returns it with its formatted line. */
    add(text) {
      const elapsedMs = Date.now() - startedAt;
      const line = `[${fmtElapsed(elapsedMs)}] ${text}`;
      const segment = { elapsedMs, text, line };
      segments.push(segment);
      if (!closed) logStream.write(line + '\n');
      return segment;
    },

    all() {
      return segments.slice();
    },

    /** Segments finalized within the last `minutes` minutes. */
    lastMinutes(minutes) {
      const cutoff = Date.now() - startedAt - minutes * 60 * 1000;
      return segments.filter((s) => s.elapsedMs >= cutoff);
    },

    /** The last `minutes` minutes of transcript, formatted as caption lines. */
    windowText(minutes) {
      return this.lastMinutes(minutes)
        .map((s) => s.line)
        .join('\n');
    },

    /**
     * Resolves once buffered lines are actually on disk. Callers MUST await
     * this before process.exit(), or the tail of the session log (the demo's
     * ground-truth artifact) gets truncated.
     */
    close() {
      if (closed) return Promise.resolve();
      closed = true;
      return new Promise((resolve) => {
        logStream.end(() => resolve());
      });
    },
  };
}

module.exports = { createTranscript, fmtElapsed };
