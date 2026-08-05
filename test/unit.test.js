// Unit tests for rcli-meet's pure logic (no models, no audio needed):
//   node --test test/
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {
  fitLinesFromEnd,
  buildPrompt,
  createThinkFilter,
  visibleOutside,
  CONTEXT_TOKEN_BUDGET,
  ANSWER_MAX_TOKENS,
} = require('../src/llm');
const { createTranscript, fmtElapsed } = require('../src/transcript');
const { createRetrieval, dot } = require('../src/retrieval');
const { assertModelPresent } = require('../src/stt');

// --- context budget -------------------------------------------------------

test('fitLinesFromEnd keeps the most recent lines that fit', () => {
  const lines = ['aaaa', 'bbbb', 'cccc'];
  // Each line costs length+1 (newline) = 5. Budget 10 fits exactly two.
  const { text, dropped } = fitLinesFromEnd(lines, 10);
  assert.strictEqual(text, 'bbbb\ncccc');
  assert.strictEqual(dropped, 1);
});

test('fitLinesFromEnd keeps everything when it fits', () => {
  const { text, dropped } = fitLinesFromEnd(['a', 'b'], 1000);
  assert.strictEqual(text, 'a\nb');
  assert.strictEqual(dropped, 0);
});

test('fitLinesFromEnd handles an empty list and a zero budget', () => {
  assert.deepStrictEqual(fitLinesFromEnd([], 100), { text: '', dropped: 0 });
  assert.deepStrictEqual(fitLinesFromEnd(['aaa'], 0), { text: '', dropped: 1 });
});

test('buildPrompt bounds a huge transcript to the context budget', () => {
  // Simulate ~40 minutes of dense captions -- far more than n_ctx=2048 allows.
  const recentLines = Array.from(
    { length: 2000 },
    (_, i) => `[00:${String(i % 60).padStart(2, '0')}:00] this is caption line number ${i}`
  );
  const prompt = buildPrompt({ recentLines, retrievedLines: [], question: 'what happened?' });

  // The whole point: the prompt must stay inside the model's context window.
  // ~3.5 chars/token, and the answer needs room too.
  const estimatedTokens = prompt.length / 3.5;
  assert.ok(
    estimatedTokens < 2048 - 200,
    `prompt should fit in a 2048-token context with room for a 200-token answer, ` +
      `estimated ${Math.round(estimatedTokens)} tokens (${prompt.length} chars)`
  );
  assert.ok(estimatedTokens <= CONTEXT_TOKEN_BUDGET + 200, 'should respect the configured budget');
});

test('buildPrompt keeps the newest captions and says what it dropped', () => {
  const recentLines = Array.from({ length: 2000 }, (_, i) => `[00:00:00] caption ${i}`);
  const prompt = buildPrompt({ recentLines, retrievedLines: [], question: 'q' });

  assert.ok(prompt.includes('caption 1999'), 'must keep the most recent line');
  assert.ok(!prompt.includes('caption 0 '), 'must drop the oldest lines');
  assert.match(prompt, /line\(s\) omitted/, 'must disclose truncation rather than hide it');
});

test('buildPrompt always includes the in-flight partial utterance', () => {
  // The partial is the most likely subject of a question, so it must survive
  // even when the window is saturated.
  const recentLines = Array.from({ length: 2000 }, (_, i) => `[00:00:00] caption ${i}`);
  const prompt = buildPrompt({
    recentLines,
    retrievedLines: [],
    partial: 'THE DEADLINE IS NEXT FRIDAY',
    question: 'what is the deadline?',
  });
  assert.ok(prompt.includes('[now] THE DEADLINE IS NEXT FRIDAY'));
});

test('buildPrompt includes the question and both sections', () => {
  const prompt = buildPrompt({
    recentLines: ['[00:00:05] hello there'],
    retrievedLines: ['[00:00:01] earlier thing'],
    question: 'what was said?',
  });
  assert.ok(prompt.includes('hello there'));
  assert.ok(prompt.includes('earlier thing'));
  assert.ok(prompt.includes('what was said?'));
});

test('buildPrompt marks empty context instead of leaving blanks', () => {
  const prompt = buildPrompt({ recentLines: [], retrievedLines: [], question: 'q' });
  assert.ok(prompt.includes('(none)'));
  assert.ok(prompt.includes('(none yet)'));
});

test('context budget plus answer budget fits a 2048-token context', () => {
  // The two budgets are set independently; this is the invariant that keeps
  // llama.cpp from rejecting the prompt outright.
  const scaffolding = 80; // prompt boilerplate
  assert.ok(
    CONTEXT_TOKEN_BUDGET + ANSWER_MAX_TOKENS + scaffolding < 2048,
    `context(${CONTEXT_TOKEN_BUDGET}) + answer(${ANSWER_MAX_TOKENS}) must fit in n_ctx=2048`
  );
});

// --- reasoning-block (<think>) filtering ----------------------------------

test('visibleOutside strips a complete think block', () => {
  const { text, thinking } = visibleOutside('<think>reasoning here</think>The answer.');
  assert.strictEqual(text, 'The answer.');
  assert.strictEqual(thinking, false);
});

test('visibleOutside hides an unterminated think block', () => {
  const { text, thinking } = visibleOutside('<think>still reasoning...');
  assert.strictEqual(text, '');
  assert.strictEqual(thinking, true);
});

test('visibleOutside passes through text with no think block', () => {
  const { text, thinking } = visibleOutside('Just a plain answer.');
  assert.strictEqual(text, 'Just a plain answer.');
  assert.strictEqual(thinking, false);
});

test('think filter never leaks a partial opening tag to the terminal', () => {
  const f = createThinkFilter();
  let shown = '';
  // "<think>" arriving one character at a time must never render as "<thin".
  for (const ch of '<think>hidden</think>visible') shown += f.push(ch).delta;
  assert.strictEqual(shown, 'visible');
});

test('think filter emits only the answer across realistic token chunks', () => {
  const f = createThinkFilter();
  const chunks = ['<th', 'ink>', 'Okay, the user asks', ' about X.', '</think', '>', 'Next', ' Friday.'];
  let shown = '';
  let sawThinking = false;
  for (const c of chunks) {
    const { delta, thinking } = f.push(c);
    if (thinking) sawThinking = true;
    shown += delta;
  }
  assert.strictEqual(shown, 'Next Friday.');
  assert.ok(sawThinking, 'should report that the model was reasoning');
  assert.strictEqual(f.visibleText.trim(), 'Next Friday.');
});

test('think filter reports empty visible text when the budget ran out mid-thought', () => {
  // This is the failure we hit live: 200 tokens all consumed reasoning.
  const f = createThinkFilter();
  for (const c of ['<think>', 'reasoning that never finishes...']) f.push(c);
  assert.strictEqual(f.visibleText, '', 'caller must be able to detect "no answer produced"');
});

test('think filter handles a plain (non-reasoning) model stream', () => {
  const f = createThinkFilter();
  let shown = '';
  for (const c of ['The ', 'deadline ', 'is ', 'Friday.']) shown += f.push(c).delta;
  assert.strictEqual(shown, 'The deadline is Friday.');
});

// --- transcript -----------------------------------------------------------

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'rcli-meet-test-'));
}

test('fmtElapsed formats hours, minutes and seconds', () => {
  assert.strictEqual(fmtElapsed(0), '00:00:00');
  assert.strictEqual(fmtElapsed(65 * 1000), '00:01:05');
  assert.strictEqual(fmtElapsed(3661 * 1000), '01:01:01');
});

test('transcript records segments and writes them to the log', async () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  t.add('first line');
  t.add('second line');

  assert.strictEqual(t.all().length, 2);
  assert.match(t.all()[0].line, /^\[00:00:0\d\] first line$/);

  await t.close();
  const onDisk = fs.readFileSync(t.logPath, 'utf8');
  assert.ok(onDisk.includes('first line'));
  assert.ok(onDisk.includes('second line'));
});

test('transcript.close() flushes before resolving (log is not truncated)', async () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  // Enough volume that the write stream must buffer rather than complete
  // synchronously -- this is what used to get lost on process.exit().
  for (let i = 0; i < 5000; i++) t.add(`line number ${i} with some padding text`);
  await t.close();

  const lines = fs.readFileSync(t.logPath, 'utf8').trim().split('\n');
  assert.strictEqual(lines.length, 5000, 'every line must be on disk after close() resolves');
  assert.ok(lines[4999].includes('line number 4999'));
});

test('transcript.close() is idempotent and stops accepting writes', async () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  t.add('kept');
  await t.close();
  await t.close(); // must not throw (double shutdown path)
  t.add('after close'); // must not throw on a closed stream
  assert.ok(!fs.readFileSync(t.logPath, 'utf8').includes('after close'));
});

test('transcript windowing excludes segments older than the window', () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  const seg = t.add('old line');
  // Backdate beyond the window instead of sleeping.
  seg.elapsedMs = -10 * 60 * 1000;
  t.add('new line');

  const recent = t.lastMinutes(5).map((s) => s.text);
  assert.deepStrictEqual(recent, ['new line']);
  assert.strictEqual(t.all().length, 2, 'windowing must not discard history');
});

// --- retrieval ------------------------------------------------------------

// Deterministic stand-in for the real embedder: unit vectors so a dot product
// is a real cosine similarity.
function fakeEmbedder(map) {
  return {
    embed(text) {
      const v = map[text];
      if (!v) throw new Error(`no fake embedding for "${text}"`);
      return Float32Array.from(v);
    },
  };
}

test('dot computes the inner product', () => {
  assert.strictEqual(dot(Float32Array.from([1, 0]), Float32Array.from([1, 0])), 1);
  assert.strictEqual(dot(Float32Array.from([1, 0]), Float32Array.from([0, 1])), 0);
});

test('retrieval ranks by similarity to the query', () => {
  const embedder = fakeEmbedder({
    deadline: [1, 0],
    budget: [0, 1],
    'when is it due': [0.99, 0.14],
  });
  const r = createRetrieval(embedder);
  r.add({ text: 'deadline', line: '[00:00:01] deadline', elapsedMs: 1000 });
  r.add({ text: 'budget', line: '[00:00:02] budget', elapsedMs: 2000 });

  const hits = r.topK('when is it due', 2);
  assert.strictEqual(hits[0].text, 'deadline', 'closest vector must rank first');
  assert.strictEqual(hits.length, 2);
});

test('retrieval honors the exclude set (no duplicate context)', () => {
  const embedder = fakeEmbedder({ a: [1, 0], b: [0, 1], q: [1, 0] });
  const r = createRetrieval(embedder);
  r.add({ text: 'a', line: 'LINE_A', elapsedMs: 1 });
  r.add({ text: 'b', line: 'LINE_B', elapsedMs: 2 });

  const hits = r.topK('q', 5, new Set(['LINE_A']));
  assert.deepStrictEqual(hits.map((h) => h.line), ['LINE_B']);
});

test('retrieval survives an embedder failure instead of crashing', () => {
  const errors = [];
  const embedder = {
    embed(text) {
      if (text === 'boom') throw new Error('native embed failed');
      return Float32Array.from([1, 0]);
    },
  };
  const r = createRetrieval(embedder, { onError: (m) => errors.push(m) });

  assert.strictEqual(r.add({ text: 'boom', line: 'L1', elapsedMs: 1 }), false);
  assert.strictEqual(r.size, 0);
  assert.strictEqual(errors.length, 1);
  // A good segment still indexes afterwards.
  assert.strictEqual(r.add({ text: 'fine', line: 'L2', elapsedMs: 2 }), true);
  assert.strictEqual(r.size, 1);
});

test('retrieval returns nothing (not a throw) when the query cannot embed', () => {
  const errors = [];
  const embedder = {
    embed(text) {
      if (text === 'bad query') throw new Error('nope');
      return Float32Array.from([1, 0]);
    },
  };
  const r = createRetrieval(embedder, { onError: (m) => errors.push(m) });
  r.add({ text: 'x', line: 'L', elapsedMs: 1 });
  assert.deepStrictEqual(r.topK('bad query'), []);
  assert.strictEqual(errors.length, 1);
});

test('retrieval on an empty index returns an empty list', () => {
  const r = createRetrieval(fakeEmbedder({}));
  assert.deepStrictEqual(r.topK('anything'), []);
});

// --- stt model validation -------------------------------------------------

test('assertModelPresent throws an actionable error when files are missing', () => {
  const dir = tmpDir();
  assert.throws(() => assertModelPresent(dir), (err) => {
    assert.match(err.message, /model files missing/);
    assert.match(err.message, /encoder-epoch-99/, 'names the missing file');
    assert.match(err.message, /curl -L/, 'tells the user how to fix it');
    return true;
  });
});

test('assertModelPresent passes when every file exists', () => {
  const dir = tmpDir();
  for (const f of [
    'encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
    'decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
    'joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
    'tokens.txt',
  ]) {
    fs.writeFileSync(path.join(dir, f), 'x');
  }
  assert.doesNotThrow(() => assertModelPresent(dir));
});
