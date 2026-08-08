// Spoken command recognition with heavy ASR-typo tolerance.
// Whisper often turns "start" into shark/stark/etc on short accented utterances.
// We only match when the WHOLE utterance is the command (1–3 short tokens),
// never because a command word appears inside a longer phrase.
const START_CANON = 'start';
const SAVE_CANON = 'save';
const LOAD_CANON = 'load';
const STOP_CANON = 'stop';

const START_ALIASES = new Set([
  'start',
  'starts',
  'started',
  'starting',
  'startstart',
  'startrecording',
  'record',
  'records',
  'recording',
  'recorded',
  'begin',
  'began',
  'begun',
  'beginning',
  // Whisper mishears of "start"
  'shark',
  'sharks',
  'sharkss',
  'stark',
  'starks',
  'sstark',
  'sttart',
  'startr',
  'starrt',
  'statt',
  'stard',
  'starte',
  'strat',
  'sart',
  'sartt',
]);

const SAVE_ALIASES = new Set(['save', 'saves', 'saved', 'saving', 'safe', 'saif', 'sayve', 'seve', 'sabe']);
const LOAD_ALIASES = new Set(['load', 'loads', 'loaded', 'loading', 'lode', 'lowed', 'lod']);
const STOP_ALIASES = new Set(['stop', 'stops', 'stopped', 'stopping']);

// Consonant skeletons that uniquely lean "start" (not save/stop/load).
const START_SKELETONS = new Set(['strt', 'strk', 'shrk', 'strtstrt', 'rcrd', 'bgn', 'bgnt']);

function normalize(text) {
  return String(text || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim()
    .replace(/^\//, '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Whole-utterance forms only. Do NOT emit each word separately -- that made
 * "I said start" / "please save it" falsely match.
 */
function commandCandidates(norm) {
  if (!norm) return [];
  const parts = norm.split(' ').filter(Boolean);
  const dedup = [];
  for (const p of parts) {
    if (dedup[dedup.length - 1] !== p) dedup.push(p);
  }
  const joined = dedup.join(' ');
  const smashed = dedup.join('');
  const out = new Set([joined, smashed]);
  // Allow the single-token form only when the utterance truly is one token
  // (or two identical tokens that deduped to one).
  if (dedup.length === 1) out.add(dedup[0]);
  return [...out];
}

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const row = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) row[j] = j;
  for (let i = 1; i <= a.length; i++) {
    let prev = i - 1;
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j];
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + cost);
      prev = tmp;
    }
  }
  return row[b.length];
}

function skeleton(s) {
  return s
    .replace(/[^a-z]/g, '')
    .replace(/([a-z])\1+/g, '$1')
    .replace(/[aeiou]/g, '');
}

function maxEdits(word) {
  if (word.length <= 3) return 0;
  if (word.length === 4) return 1;
  return 2;
}

function fuzzyHit(candidate, aliases, canon) {
  if (!candidate) return false;
  const compact = candidate.replace(/\s+/g, '');
  if (aliases.has(candidate) || aliases.has(compact)) return true;
  if (candidate.includes(' ')) return false; // multi-word: exact alias/smash only
  if (levenshtein(candidate, canon) <= maxEdits(canon)) return true;
  for (const alias of aliases) {
    if (alias.length < 4) continue;
    if (levenshtein(candidate, alias) <= maxEdits(alias)) return true;
  }
  return false;
}

function isCommandShaped(norm) {
  const parts = norm.split(' ').filter(Boolean);
  if (parts.length === 0 || parts.length > 3) return false;
  if (norm.length > 32) return false;
  return true;
}

/**
 * @param allowStop {boolean} false for spoken lines
 * @returns {'start'|'save'|'load'|'stop'|null}
 */
function parseCommand(text, { allowStop = true } = {}) {
  const norm = normalize(text);
  if (!norm || !isCommandShaped(norm)) return null;

  const candidates = commandCandidates(norm);

  for (const c of candidates) {
    if (fuzzyHit(c, START_ALIASES, START_CANON)) return 'start';
    if (!c.includes(' ') && START_SKELETONS.has(skeleton(c))) return 'start';
  }
  for (const c of candidates) {
    if (fuzzyHit(c, SAVE_ALIASES, SAVE_CANON)) return 'save';
  }
  for (const c of candidates) {
    if (fuzzyHit(c, LOAD_ALIASES, LOAD_CANON)) return 'load';
  }
  if (allowStop) {
    for (const c of candidates) {
      if (fuzzyHit(c, STOP_ALIASES, STOP_CANON)) return 'stop';
    }
  }
  return null;
}

module.exports = {
  parseCommand,
  normalize,
  levenshtein,
  skeleton,
  START_ALIASES,
};
