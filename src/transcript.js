// Rolling transcript: timestamped segments in memory, appended live to a
// per-session log file (the scrollback "ground truth" for every LLM answer).
//
// Segments carry a `source` ('meeting' or 'you') since two independent audio
// streams feed this. Two streams finalize on independent timers, so a "you"
// segment from 5s ago can arrive after a "meeting" segment from just now --
// sorting by elapsedMs on read (not relying on insertion order) keeps the
// transcript in real chronological order regardless of arrival order.
const fs = require('fs');
const path = require('path');

const SOURCE_LABEL = { meeting: 'meeting', you: 'you' };

function fmtElapsed(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = String(Math.floor(totalSec / 3600)).padStart(2, '0');
  const m = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
  const s = String(totalSec % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function byElapsed(a, b) {
  return a.elapsedMs - b.elapsedMs;
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

    /**
     * Append a finalized caption segment; returns it with its formatted line.
     * @param source {'meeting'|'you'}
     */
    add(text, source) {
      if (!SOURCE_LABEL[source]) {
        throw new Error(`transcript.add: source must be "meeting" or "you", got "${source}"`);
      }
      const elapsedMs = Date.now() - startedAt;
      const line = `[${fmtElapsed(elapsedMs)}] [${SOURCE_LABEL[source]}] ${text}`;
      const segment = { elapsedMs, text, source, line };
      segments.push(segment);
      // Written in append order rather than sorted order -- interleaving is
      // rare enough (only near-simultaneous cross-talk) that a strictly
      // arrival-ordered log is more useful for debugging than a
      // sorted-but-rewritten one, and the log is append-only by design.
      if (!closed) logStream.write(line + '\n');
      return segment;
    },

    all() {
      return segments.slice().sort(byElapsed);
    },

    /** Segments finalized within the last `minutes` minutes, chronological. */
    lastMinutes(minutes) {
      const cutoff = Date.now() - startedAt - minutes * 60 * 1000;
      return segments.filter((s) => s.elapsedMs >= cutoff).sort(byElapsed);
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
