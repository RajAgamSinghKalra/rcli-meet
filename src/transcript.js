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
const OVERLAP_SUFFIX = '  (overlapping speech -- talked over)';

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

/** Do [a.startElapsedMs, a.elapsedMs] and [b.startElapsedMs, b.elapsedMs] intersect? */
function rangesOverlap(a, b) {
  return a.startElapsedMs < b.elapsedMs && b.startElapsedMs < a.elapsedMs;
}

/**
 * Cross-talk (both audio sources active at once) means two segments were
 * genuinely simultaneous, not sequential -- worth telling the model, since it
 * changes both "who replied to what" reasoning and how much to trust the
 * transcription (overlapping audio degrades ASR accuracy). Only flags
 * different-source neighbors: two people from the SAME source can't overlap
 * (one mic/one loopback stream each), so same-source adjacency is never
 * marked.
 */
function withOverlapMarkers(sorted) {
  return sorted.map((seg, i) => {
    const prev = sorted[i - 1];
    const next = sorted[i + 1];
    const overlaps =
      (prev && prev.source !== seg.source && rangesOverlap(seg, prev)) ||
      (next && next.source !== seg.source && rangesOverlap(seg, next));
    return overlaps ? { ...seg, line: seg.line + OVERLAP_SUFFIX } : seg;
  });
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

    /** Elapsed ms since session start, on the same clock as every segment's timestamps. */
    elapsedNow() {
      return Date.now() - startedAt;
    },

    /**
     * Append a finalized caption segment; returns it with its formatted line.
     * @param source {'meeting'|'you'}
     * @param startElapsedMs {number} when this utterance began (for overlap
     *   detection against the other source); defaults to the finalize time,
     *   which disables overlap detection for that segment.
     */
    add(text, source, startElapsedMs) {
      if (!SOURCE_LABEL[source]) {
        throw new Error(`transcript.add: source must be "meeting" or "you", got "${source}"`);
      }
      const elapsedMs = Date.now() - startedAt;
      const line = `[${fmtElapsed(elapsedMs)}] [${SOURCE_LABEL[source]}] ${text}`;
      const segment = {
        elapsedMs,
        startElapsedMs: startElapsedMs ?? elapsedMs,
        text,
        source,
        line,
      };
      segments.push(segment);
      // Written in append order rather than sorted order -- interleaving is
      // rare enough (only near-simultaneous cross-talk) that a strictly
      // arrival-ordered log is more useful for debugging than a
      // sorted-but-rewritten one, and the log is append-only by design.
      if (!closed) logStream.write(line + '\n');
      return segment;
    },

    all() {
      return withOverlapMarkers(segments.slice().sort(byElapsed));
    },

    /** Segments finalized within the last `minutes` minutes, chronological. */
    lastMinutes(minutes) {
      const cutoff = Date.now() - startedAt - minutes * 60 * 1000;
      const sorted = segments.slice().sort(byElapsed);
      // Overlap markers need neighbors outside the window too (a segment at
      // the window's edge might overlap one just before it), so mark against
      // the full sorted list, then filter.
      return withOverlapMarkers(sorted).filter((s) => s.elapsedMs >= cutoff);
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

module.exports = { createTranscript, fmtElapsed, withOverlapMarkers, OVERLAP_SUFFIX };
