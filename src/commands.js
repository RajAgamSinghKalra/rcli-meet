// Recognizes start/save/load/stop regardless of whether they arrived typed
// or spoken. "stop" is deliberately typed-only (see allowStop) -- it's a
// common word in ordinary conversation, and triggering session-stop on every
// mid-sentence "stop" would be far worse than requiring it be typed.
const START_WORDS = new Set(['start', 'record', 'begin', 'start recording']);
const SAVE_WORDS = new Set(['save']);
const LOAD_WORDS = new Set(['load']);
const STOP_WORDS = new Set(['stop']);

function normalize(text) {
  return String(text)
    .trim()
    .replace(/^\//, '')
    .toLowerCase()
    .replace(/[.!?,]+$/, '');
}

/**
 * @param allowStop {boolean} pass false for spoken utterances -- only a
 *   typed line should ever be able to stop the session.
 * @returns {'start'|'save'|'load'|'stop'|null}
 */
function parseCommand(text, { allowStop = true } = {}) {
  const norm = normalize(text);
  if (START_WORDS.has(norm)) return 'start';
  if (SAVE_WORDS.has(norm)) return 'save';
  if (LOAD_WORDS.has(norm)) return 'load';
  if (allowStop && STOP_WORDS.has(norm)) return 'stop';
  return null;
}

module.exports = { parseCommand, normalize };
