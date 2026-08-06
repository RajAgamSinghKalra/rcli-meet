// Save/load persistence: a saved session is a folder of plain files (a
// transcript log, a summary, and copies of any ingested files) -- not raw
// audio. Re-loadable and searchable is what the Q&A use case actually needs;
// audio playback isn't.
//
// Stateless by design: main.js owns the "is a session currently active" /
// "which directory" state itself (that state has to survive stop, which
// this module has no reason to know about), so this just does filesystem work.
const fs = require('fs');
const path = require('path');

const DEFAULT_SESSIONS_DIR = path.join(__dirname, '..', 'sessions');

function slugify(s) {
  return (
    String(s)
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 60) || 'session'
  );
}

function isoStamp(date) {
  return date.toISOString().replace(/:/g, '-').replace(/\..+/, '');
}

/** A fresh, not-yet-created directory name for a new recording session. */
function newSessionDir(baseDir, appName, now = new Date()) {
  return path.join(baseDir, `${isoStamp(now)}_${slugify(appName || 'unknown-app')}`);
}

/** Copies the transcript log + writes summary/metadata into `dir`. */
function saveSession(dir, { transcriptLogPath, summary = '', appName = 'unknown-app', files = [] } = {}) {
  fs.mkdirSync(path.join(dir, 'files'), { recursive: true });
  if (transcriptLogPath && fs.existsSync(transcriptLogPath)) {
    fs.copyFileSync(transcriptLogPath, path.join(dir, 'transcript.log'));
  }
  fs.writeFileSync(path.join(dir, 'summary.txt'), summary);
  fs.writeFileSync(
    path.join(dir, 'meta.json'),
    JSON.stringify({ appName, savedAt: new Date().toISOString(), files }, null, 2)
  );
  return dir;
}

/** Copies a file into a session's files/ dir; returns the new path. */
function addFileToSession(dir, srcPath) {
  if (!fs.existsSync(srcPath)) throw new Error(`no such file: ${srcPath}`);
  fs.mkdirSync(path.join(dir, 'files'), { recursive: true });
  const destName = path.basename(srcPath);
  const destPath = path.join(dir, 'files', destName);
  fs.copyFileSync(srcPath, destPath);
  return destName;
}

/** Saved session directory names under `baseDir`, newest first. */
function listSessions(baseDir = DEFAULT_SESSIONS_DIR) {
  if (!fs.existsSync(baseDir)) return [];
  return fs
    .readdirSync(baseDir)
    .filter((name) => fs.statSync(path.join(baseDir, name)).isDirectory())
    .sort()
    .reverse();
}

/** Reads back everything needed to seed context from a saved session. */
function loadSession(baseDir, name) {
  const dir = path.join(baseDir, name);
  if (!fs.existsSync(dir)) throw new Error(`no saved session named "${name}"`);
  const transcriptPath = path.join(dir, 'transcript.log');
  const summaryPath = path.join(dir, 'summary.txt');
  const filesDir = path.join(dir, 'files');
  return {
    dir,
    transcriptText: fs.existsSync(transcriptPath) ? fs.readFileSync(transcriptPath, 'utf8') : '',
    summary: fs.existsSync(summaryPath) ? fs.readFileSync(summaryPath, 'utf8') : '',
    files: fs.existsSync(filesDir)
      ? fs.readdirSync(filesDir).map((f) => ({ name: f, text: fs.readFileSync(path.join(filesDir, f), 'utf8') }))
      : [],
  };
}

module.exports = {
  slugify,
  newSessionDir,
  saveSession,
  addFileToSession,
  listSessions,
  loadSession,
  DEFAULT_SESSIONS_DIR,
};
