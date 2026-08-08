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
  supportsNoThink,
  serialize,
  NO_THINK_DIRECTIVE,
  TRANSCRIPTION_CAVEAT,
  CONTEXT_TOKEN_BUDGET,
  ANSWER_MAX_TOKENS,
} = require('../src/llm');
const { createTranscript, fmtElapsed } = require('../src/transcript');
const { createRetrieval, dot } = require('../src/retrieval');
const { assertModelPresent } = require('../src/stt');
const { createSummarizer, buildSummaryPrompt } = require('../src/summary');

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
  const estimatedTokens = prompt.length / 3.5;

  // The invariant that actually matters -- not an approximation of internal
  // overhead (instructions, section headers, the truncation note itself all
  // legitimately vary), but the one thing that must always hold: prompt +
  // answer budget fits under the model's context.
  assert.ok(
    estimatedTokens + ANSWER_MAX_TOKENS < 2048,
    `full prompt (~${Math.round(estimatedTokens)} tokens) + answer budget (${ANSWER_MAX_TOKENS}) must fit n_ctx=2048`
  );
});

test('buildPrompt keeps the newest captions and says what it dropped', () => {
  const recentLines = Array.from({ length: 2000 }, (_, i) => `[00:00:00] caption ${i}`);
  const prompt = buildPrompt({ recentLines, retrievedLines: [], question: 'q' });

  assert.ok(prompt.includes('caption 1999'), 'must keep the most recent line');
  assert.ok(!prompt.includes('caption 0 '), 'must drop the oldest lines');
  assert.match(prompt, /line\(s\) omitted/, 'must disclose truncation rather than hide it');
});

test('buildPrompt always includes in-flight partials from both sources', () => {
  // Partials are the most likely subject of a question, so they must survive
  // even when the window is saturated.
  const recentLines = Array.from({ length: 2000 }, (_, i) => `[00:00:00] [meeting] caption ${i}`);
  const prompt = buildPrompt({
    recentLines,
    retrievedLines: [],
    partials: { meeting: 'THE DEADLINE IS NEXT FRIDAY', you: 'GOT IT THANKS' },
    question: 'what is the deadline?',
  });
  assert.ok(prompt.includes('[now] [meeting] THE DEADLINE IS NEXT FRIDAY'));
  assert.ok(prompt.includes('[now] [you] GOT IT THANKS'));
});

test('buildPrompt omits a partial section for a source with nothing in-flight', () => {
  const prompt = buildPrompt({ partials: { meeting: 'HELLO' }, question: 'q' });
  assert.ok(prompt.includes('[now] [meeting] HELLO'));
  assert.ok(!prompt.includes('[now] [you]'));
});

test('buildPrompt tells the model the transcript has ASR errors to interpret past', () => {
  const prompt = buildPrompt({ question: 'q' });
  assert.ok(prompt.includes(TRANSCRIPTION_CAVEAT));
  assert.match(prompt, /similar-sounding/i);
});

test('buildSummaryPrompt also carries the transcription-error caveat', () => {
  const prompt = buildSummaryPrompt('s', ['[meeting] hi'], false);
  assert.ok(prompt.includes(TRANSCRIPTION_CAVEAT));
});

test('buildPrompt teaches reply adjacency and overlap handling', () => {
  const prompt = buildPrompt({ question: 'q' });
  assert.match(prompt, /chronological order/i);
  assert.match(prompt, /direct reply/i);
  assert.match(prompt, /overlapping speech.*talked over/i);
  assert.match(prompt, /simultaneously/i);
});

test('buildPrompt explains the meeting/you distinction to the model', () => {
  const prompt = buildPrompt({ question: 'q' });
  assert.match(prompt, /\[meeting\][\s\S]*other people/i);
  assert.match(prompt, /\[you\][\s\S]*THE USER/i);
});

test('buildPrompt includes the rolling summary, capped, ahead of the recency window', () => {
  const prompt = buildPrompt({ summary: 'They discussed the Q3 roadmap.', question: 'q' });
  assert.ok(prompt.includes('They discussed the Q3 roadmap.'));
  assert.ok(!prompt.includes('no summary yet'));

  const empty = buildPrompt({ question: 'q' });
  assert.ok(empty.includes('no summary yet'));
});

test('buildPrompt reserves context space for the summary before the recency window', () => {
  const longSummary = 'S'.repeat(2000);
  const recentLines = Array.from({ length: 2000 }, (_, i) => `[00:00:00] [meeting] caption ${i}`);
  const prompt = buildPrompt({ summary: longSummary, recentLines, question: 'q' });
  const estimatedTokens = prompt.length / 3.5;
  assert.ok(
    estimatedTokens + ANSWER_MAX_TOKENS < 2048,
    `summary + transcript + answer together must still fit n_ctx=2048, got ~${Math.round(estimatedTokens)} prompt tokens`
  );
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

// --- suppressing reasoning at the source ----------------------------------

test('supportsNoThink detects Qwen models by id or path', () => {
  assert.strictEqual(supportsNoThink('qwen2.5-3b'), true);
  assert.strictEqual(supportsNoThink('D:/models/Qwen3-4B-Q4_K_M.gguf'), true);
  assert.strictEqual(supportsNoThink('llama-3.2-3b'), false);
  assert.strictEqual(supportsNoThink(undefined), false);
});

test('buildPrompt appends the no-think directive only when asked', () => {
  const withNoThink = buildPrompt({ question: 'q', disableThinking: true });
  assert.ok(withNoThink.trimEnd().endsWith(NO_THINK_DIRECTIVE));

  const withThink = buildPrompt({ question: 'q', disableThinking: false });
  assert.ok(!withThink.includes(NO_THINK_DIRECTIVE));
  // Default must not silently change model behavior.
  assert.ok(!buildPrompt({ question: 'q' }).includes(NO_THINK_DIRECTIVE));
});

test('the no-think directive costs almost nothing against the budget', () => {
  const base = buildPrompt({ question: 'q', disableThinking: false });
  const withDirective = buildPrompt({ question: 'q', disableThinking: true });
  assert.ok(withDirective.length - base.length <= NO_THINK_DIRECTIVE.length + 2);
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

test('an empty think block (Qwen3 under /no_think) reports no real reasoning', () => {
  // Qwen3 still emits <think>\n\n</think> when reasoning is disabled; that
  // must not flash a "thinking..." indicator at the user.
  const { thinkingChars } = visibleOutside('<think>\n\n</think>Next Friday.');
  assert.ok(thinkingChars <= 8, `empty block should have ~no content, got ${thinkingChars}`);
  assert.strictEqual(visibleOutside('<think>\n\n</think>Next Friday.').text, 'Next Friday.');
});

test('a substantive think block reports real reasoning content', () => {
  const { thinkingChars } = visibleOutside(
    '<think>Okay, the user is asking about the deadline, let me check.</think>Friday.'
  );
  assert.ok(thinkingChars > 8, 'real reasoning must be distinguishable from an empty block');
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
  t.add('first line', 'meeting');
  t.add('second line', 'you');

  assert.strictEqual(t.all().length, 2);
  assert.match(t.all()[0].line, /^\[00:00:0\d\] \[meeting\] first line$/);
  assert.match(t.all()[1].line, /^\[00:00:0\d\] \[you\] second line$/);

  await t.close();
  const onDisk = fs.readFileSync(t.logPath, 'utf8');
  assert.ok(onDisk.includes('first line'));
  assert.ok(onDisk.includes('second line'));
});

test('transcript.add rejects an unknown source', () => {
  const t = createTranscript(tmpDir());
  assert.throws(() => t.add('x', 'bogus'), /source must be/);
});

test('transcript sorts by time across two independently-arriving sources', () => {
  // Two independent STT streams finalize on their own timers -- a "you"
  // segment from 5s ago can arrive after a "meeting" segment from just now.
  // Reading must reflect real chronological order, not arrival order.
  const t = createTranscript(tmpDir());
  const early = t.add('said first', 'you');
  early.elapsedMs = 1000;
  const late = t.add('said second', 'meeting'); // arrives second but is earlier... no, later
  late.elapsedMs = 5000;
  // Now simulate an out-of-order arrival: a "you" segment whose actual speech
  // time is BETWEEN the two above, but which finalizes last.
  const middle = t.add('said in between', 'you');
  middle.elapsedMs = 3000;

  const ordered = t.all().map((s) => s.text);
  assert.deepStrictEqual(ordered, ['said first', 'said in between', 'said second']);
});

// --- cross-talk / overlap detection ----------------------------------------

const { withOverlapMarkers, OVERLAP_SUFFIX } = require('../src/transcript');

function seg(source, startElapsedMs, elapsedMs, text = source) {
  return { source, startElapsedMs, elapsedMs, text, line: `[00:00:00] [${source}] ${text}` };
}

test('withOverlapMarkers flags genuinely simultaneous cross-source speech', () => {
  // "you" talks 1000-3000ms, "meeting" talks 2000-4000ms -- ranges overlap.
  const marked = withOverlapMarkers([seg('you', 1000, 3000), seg('meeting', 2000, 4000)]);
  assert.ok(marked[0].line.endsWith(OVERLAP_SUFFIX));
  assert.ok(marked[1].line.endsWith(OVERLAP_SUFFIX));
});

test('withOverlapMarkers does not flag sequential (non-overlapping) turns', () => {
  const marked = withOverlapMarkers([seg('meeting', 0, 1000), seg('you', 1500, 2000)]);
  assert.ok(!marked[0].line.endsWith(OVERLAP_SUFFIX));
  assert.ok(!marked[1].line.endsWith(OVERLAP_SUFFIX));
});

test('withOverlapMarkers never flags two segments from the SAME source', () => {
  // One mic, one loopback stream -- a source can't overlap with itself.
  const marked = withOverlapMarkers([seg('you', 0, 3000), seg('you', 1000, 2000)]);
  assert.ok(!marked[0].line.endsWith(OVERLAP_SUFFIX));
  assert.ok(!marked[1].line.endsWith(OVERLAP_SUFFIX));
});

test('transcript.add records a real start time for overlap detection', () => {
  const t = createTranscript(tmpDir());
  const s = t.add('hi', 'meeting', 42);
  assert.strictEqual(s.startElapsedMs, 42);
});

test('transcript.lastMinutes marks overlap using neighbors outside the window too', () => {
  const t = createTranscript(tmpDir());
  const old = t.add('old overlapping', 'meeting', 0);
  old.elapsedMs = -10 * 60 * 1000; // outside a 5-minute window
  old.startElapsedMs = -10 * 60 * 1000 - 500;
  const recent = t.add('recent', 'you', old.elapsedMs - 100); // overlaps `old`
  recent.startElapsedMs = old.elapsedMs - 100;

  const windowed = t.lastMinutes(5);
  assert.strictEqual(windowed.length, 1, 'the old segment itself should be excluded');
  assert.ok(windowed[0].line.endsWith(OVERLAP_SUFFIX), 'but still marked from its out-of-window neighbor');
});

test('transcript.close() flushes before resolving (log is not truncated)', async () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  // Enough volume that the write stream must buffer rather than complete
  // synchronously -- this is what used to get lost on process.exit().
  for (let i = 0; i < 5000; i++) t.add(`line number ${i} with some padding text`, 'meeting');
  await t.close();

  const lines = fs.readFileSync(t.logPath, 'utf8').trim().split('\n');
  assert.strictEqual(lines.length, 5000, 'every line must be on disk after close() resolves');
  assert.ok(lines[4999].includes('line number 4999'));
});

test('transcript.close() is idempotent and stops accepting writes', async () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  t.add('kept', 'meeting');
  await t.close();
  await t.close(); // must not throw (double shutdown path)
  t.add('after close', 'meeting'); // must not throw on a closed stream
  assert.ok(!fs.readFileSync(t.logPath, 'utf8').includes('after close'));
});

test('transcript windowing excludes segments older than the window', () => {
  const dir = tmpDir();
  const t = createTranscript(dir);
  const seg = t.add('old line', 'meeting');
  // Backdate beyond the window instead of sleeping.
  seg.elapsedMs = -10 * 60 * 1000;
  t.add('new line', 'you');

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

// --- LLM call serialization ------------------------------------------------

test('serialize never runs two functions concurrently', async () => {
  // llama.cpp contexts aren't safe for concurrent decode: a summary update
  // racing a question's generate() call on the same context is the failure
  // this exists to prevent. Simulate overlapping async work and assert the
  // "busy" flag is never true when a second call starts.
  let busy = false;
  let overlapDetected = false;
  const order = [];

  function slowTask(id, ms) {
    return serialize(async () => {
      if (busy) overlapDetected = true;
      busy = true;
      order.push(`start:${id}`);
      await new Promise((r) => setTimeout(r, ms));
      order.push(`end:${id}`);
      busy = false;
    });
  }

  const a = slowTask('A', 30);
  const b = slowTask('B', 5);
  const c = slowTask('C', 5);
  await Promise.all([a, b, c]);

  assert.strictEqual(overlapDetected, false, 'no two tasks should ever run concurrently');
  // Queued in call order: A fully finishes before B starts, B before C.
  assert.deepStrictEqual(order, ['start:A', 'end:A', 'start:B', 'end:B', 'start:C', 'end:C']);
});

test('serialize continues the queue after a task throws', async () => {
  const results = [];
  await serialize(async () => {
    throw new Error('boom');
  }).catch(() => results.push('caught first'));
  await serialize(async () => {
    results.push('second ran');
  });
  assert.deepStrictEqual(results, ['caught first', 'second ran']);
});

// --- rolling meeting summary ------------------------------------------------

function fakeLLM(responder) {
  return {
    async *generate(prompt) {
      for (const tok of responder(prompt)) yield tok;
    },
  };
}

test('summarizer does nothing until enough segments accumulate', () => {
  const llm = fakeLLM(() => {
    throw new Error('should not be called yet');
  });
  const s = createSummarizer({ llm });
  for (let i = 0; i < 7; i++) s.addSegment({ line: `line ${i}` });
  s.maybeUpdate(); // below the default threshold of 8
  assert.strictEqual(s.summary, '');
  assert.strictEqual(s.pendingCount, 7);
});

test('summarizer folds new segments into the summary once the threshold is hit', async () => {
  const llm = fakeLLM((prompt) => {
    assert.ok(prompt.includes('line 0'), 'new lines must reach the prompt');
    return ['Updated: ', 'discussed the roadmap.'];
  });
  const s = createSummarizer({ llm });
  for (let i = 0; i < 8; i++) s.addSegment({ line: `line ${i}` });
  s.maybeUpdate();

  await new Promise((r) => setTimeout(r, 20)); // let the fire-and-forget settle
  assert.strictEqual(s.summary, 'Updated: discussed the roadmap.');
  assert.strictEqual(s.pendingCount, 0, 'folded segments must be cleared');
});

test('summarizer strips a <think> block from its own output', async () => {
  const llm = fakeLLM(() => ['<think>reasoning</think>', 'The short summary.']);
  const s = createSummarizer({ llm });
  for (let i = 0; i < 8; i++) s.addSegment({ line: `line ${i}` });
  s.maybeUpdate();
  await new Promise((r) => setTimeout(r, 20));
  assert.strictEqual(s.summary, 'The short summary.');
});

test('summarizer restores pending segments if the update fails', async () => {
  const errors = [];
  const llm = fakeLLM(() => {
    throw new Error('generation failed');
  });
  const s = createSummarizer({ llm, onError: (m) => errors.push(m) });
  for (let i = 0; i < 8; i++) s.addSegment({ line: `line ${i}` });
  s.maybeUpdate();
  await new Promise((r) => setTimeout(r, 20));
  assert.strictEqual(errors.length, 1);
  assert.strictEqual(s.pendingCount, 8, 'content must not be silently lost on failure');
});

test('buildSummaryPrompt asks the model to preserve the meeting/you distinction', () => {
  const prompt = buildSummaryPrompt('existing', ['[meeting] hi', '[you] hello'], false);
  assert.ok(prompt.includes('existing'));
  assert.ok(prompt.includes('[meeting] hi'));
  assert.match(prompt, /\[you\][\s\S]*\[meeting\]|distinction/i);
});

test("buildSummaryPrompt appends /no_think when requested", () => {
  const withDirective = buildSummaryPrompt('s', ['l'], true);
  assert.ok(withDirective.trimEnd().endsWith(NO_THINK_DIRECTIVE));
  assert.ok(!buildSummaryPrompt('s', ['l'], false).includes(NO_THINK_DIRECTIVE));
});

test('summarizer and askQuestion share the same lock (never overlap)', async () => {
  // The property that actually matters end-to-end: a summary update and a
  // question answer must never run generate() at the same time even though
  // they're triggered from completely different call sites.
  const { askQuestion } = require('../src/llm');
  let busy = false;
  let overlapDetected = false;

  const llm = {
    async *generate() {
      if (busy) overlapDetected = true;
      busy = true;
      yield 'tok1';
      await new Promise((r) => setTimeout(r, 15));
      yield 'tok2';
      busy = false;
    },
  };

  const s = createSummarizer({ llm });
  for (let i = 0; i < 8; i++) s.addSegment({ line: `line ${i}` });

  const questionPromise = askQuestion(llm, { question: 'q' }, () => {});
  s.maybeUpdate(); // fires right after, on the same llm
  await questionPromise;
  await new Promise((r) => setTimeout(r, 30));

  assert.strictEqual(overlapDetected, false);
});

// --- capture source validation ---------------------------------------------

test('startCapture rejects an unknown source before spawning anything', () => {
  const { startCapture, filterCaptureStderr } = require('../src/capture');
  assert.throws(() => startCapture('bogus', () => {}), /must be "loopback" or "mic"/);
  assert.ok(!/discontinuity/i.test(filterCaptureStderr('data discontinuity in recording\nok\n')));
  assert.match(filterCaptureStderr('on: Speakers\n'), /on: Speakers/);
});

// --- log-noise filter (quiet.js) ------------------------------------------

const { createFilter } = require('../src/quiet');

function collect() {
  const out = [];
  const f = createFilter((t) => out.push(t));
  return { f, text: () => out.join('') };
}

test('filter drops [RAC] log lines and keeps everything else', () => {
  const { f, text } = collect();
  f.push('[RAC][INFO][LLM] loading | file=x.cpp:1\n');
  f.push('[rcli-meet] STT ready.\n');
  f.push('[RAC][WARN][Sherpa] rac_plugin_register failed: -811\n');
  f.push('[00:00:05] THE DEADLINE IS FRIDAY\n');
  assert.strictEqual(text(), '[rcli-meet] STT ready.\n[00:00:05] THE DEADLINE IS FRIDAY\n');
});

test('filter passes live caption text through IMMEDIATELY (no newline needed)', () => {
  // Captions repaint in place with no trailing newline. Buffering them until a
  // newline arrives would freeze the live display -- the whole point of the demo.
  const { f, text } = collect();
  f.push('\x1b[2K\x1b[1G… THE PROJECT');
  assert.strictEqual(text(), '\x1b[2K\x1b[1G… THE PROJECT', 'must not be held back');
  f.push(' DEADLINE');
  assert.strictEqual(text(), '\x1b[2K\x1b[1G… THE PROJECT DEADLINE');
});

test('filter passes streaming answer tokens through immediately', () => {
  const { f, text } = collect();
  f.push('>> ');
  f.push('They said ');
  f.push('next Friday.');
  assert.strictEqual(text(), '>> They said next Friday.');
});

test('filter holds a partial [RAC] line until it can be identified', () => {
  const { f, text } = collect();
  f.push('[RA'); // ambiguous prefix -- must wait
  assert.strictEqual(text(), '');
  f.push('C][INFO][X] noise\n');
  assert.strictEqual(text(), '', 'resolved to a log line, so dropped');
});

test('filter does not mistake a caption timestamp for a log prefix', () => {
  // "[00:" shares the leading "[" with "[RAC]" but is not a prefix of it.
  const { f, text } = collect();
  f.push('[00:');
  assert.strictEqual(text(), '[00:', 'caption timestamps must stream immediately');
});

test('filter handles log and live text interleaved in one chunk', () => {
  const { f, text } = collect();
  f.push('[RAC][INFO][X] a\n[rcli-meet] b\n[RAC][INFO][X] c\n… live');
  assert.strictEqual(text(), '[rcli-meet] b\n… live');
});

test('filter flush emits held non-log text', () => {
  const { f, text } = collect();
  f.push('[R');
  f.flush();
  assert.strictEqual(text(), '', 'a held log prefix stays dropped');

  const b = collect();
  b.f.push('trailing answer text');
  b.f.flush();
  assert.strictEqual(b.text(), 'trailing answer text');
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
    'encoder-epoch-99-avg-1-chunk-16-left-128.onnx',
    'decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
    'joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
    'tokens.txt',
  ]) {
    fs.writeFileSync(path.join(dir, f), 'x');
  }
  assert.doesNotThrow(() => assertModelPresent(dir));
});

// --- command parsing (start/stop/save/load, typed and spoken) -------------

const { parseCommand } = require('../src/commands');

test('parseCommand recognizes start/save/load case-insensitively, with punctuation', () => {
  assert.strictEqual(parseCommand('start'), 'start');
  assert.strictEqual(parseCommand('Start.'), 'start');
  assert.strictEqual(parseCommand('RECORD'), 'start');
  assert.strictEqual(parseCommand('save'), 'save');
  assert.strictEqual(parseCommand('load'), 'load');
});

test('parseCommand: stop requires allowStop (spoken utterances never stop the session)', () => {
  assert.strictEqual(parseCommand('stop'), 'stop'); // default allowStop: true (typed)
  assert.strictEqual(parseCommand('stop', { allowStop: false }), null); // spoken
});

test('parseCommand ignores ordinary sentences containing a command word', () => {
  // "stop" mid-sentence must not trigger -- only an utterance that IS just the word.
  assert.strictEqual(parseCommand('please stop doing that'), null);
  assert.strictEqual(parseCommand('let us start the discussion'), null);
});

test('parseCommand strips a leading slash', () => {
  assert.strictEqual(parseCommand('/start'), 'start');
});

test('parseCommand returns null for a real question', () => {
  assert.strictEqual(parseCommand('what did they say the deadline was?'), null);
});

test('parseCommand accepts common ASR mishears of start', () => {
  for (const w of ['Sharks.', 'Shark', 'stark', 'sstark', 'startstart', 'Started', 'start recording']) {
    assert.strictEqual(parseCommand(w, { allowStop: false }), 'start', w);
  }
});

test('parseCommand accepts fuzzy near-misses via edit distance', () => {
  assert.strictEqual(parseCommand('strat', { allowStop: false }), 'start');
  assert.strictEqual(parseCommand('saave', { allowStop: false }), 'save');
});

test('parseCommand does not treat sharky questions as start', () => {
  assert.strictEqual(parseCommand('are there sharks in the ocean?', { allowStop: false }), null);
  assert.strictEqual(parseCommand('tell me about sharks', { allowStop: false }), null);
});

// --- session save/load ------------------------------------------------------

const {
  slugify,
  newSessionDir,
  saveSession,
  addFileToSession,
  listSessions,
  loadSession,
} = require('../src/session');

test('slugify produces a filesystem-safe, bounded name', () => {
  assert.strictEqual(slugify('Google Chrome - Meeting (Q3 Review)!!'), 'google-chrome-meeting-q3-review');
  assert.strictEqual(slugify(''), 'session');
});

test('newSessionDir names the folder with a date and the app name', () => {
  const dir = newSessionDir('/base', 'Zoom Meeting');
  assert.match(path.basename(dir), /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_zoom-meeting$/);
});

test('save then load round-trips the transcript, summary, and files', () => {
  const base = tmpDir();
  const dir = path.join(base, 'sess-1');
  fs.mkdirSync(dir, { recursive: true });
  const logPath = path.join(base, 'live.log');
  fs.writeFileSync(logPath, '[00:00:01] [meeting] hello\n[00:00:02] [you] hi\n');

  saveSession(dir, { transcriptLogPath: logPath, summary: 'A short summary.', appName: 'Zoom' });
  addFileToSession(dir, (() => {
    const f = path.join(base, 'notes.txt');
    fs.writeFileSync(f, 'some notes');
    return f;
  })());

  const loaded = loadSession(base, 'sess-1');
  assert.ok(loaded.transcriptText.includes('hello'));
  assert.strictEqual(loaded.summary, 'A short summary.');
  assert.strictEqual(loaded.files.length, 1);
  assert.strictEqual(loaded.files[0].text, 'some notes');

  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8'));
  assert.strictEqual(meta.appName, 'Zoom');
});

test('listSessions returns saved names newest-first, empty when none exist', () => {
  const base = tmpDir();
  assert.deepStrictEqual(listSessions(base), []);
  fs.mkdirSync(path.join(base, '2024-01-01_a'));
  fs.mkdirSync(path.join(base, '2024-06-01_b'));
  assert.deepStrictEqual(listSessions(base), ['2024-06-01_b', '2024-01-01_a']);
});

test('loadSession throws a clear error for an unknown session name', () => {
  assert.throws(() => loadSession(tmpDir(), 'nope'), /no saved session named/);
});

test('addFileToSession rejects a nonexistent source file', () => {
  assert.throws(() => addFileToSession(tmpDir(), '/no/such/file.txt'), /no such file/);
});

// --- energy VAD segmenter (sttWhisper.js) ----------------------------------

const { createEnergyVad, SAMPLE_RATE: WHISPER_SAMPLE_RATE } = require('../src/vadEnergy');
const { scrubHallucination } = require('../src/sttVulkan');

function tone(ms, amplitude) {
  return new Float32Array(Math.round((WHISPER_SAMPLE_RATE * ms) / 1000)).fill(amplitude);
}

test('energy VAD stays silent through quiet audio -- no speech start, no utterance', () => {
  const starts = [];
  const utterances = [];
  const vad = createEnergyVad({ onSpeechStart: () => starts.push(1), onUtterance: (s) => utterances.push(s) });
  for (let i = 0; i < 10; i++) vad.feed(tone(100, 0.001)); // well under the threshold
  assert.strictEqual(starts.length, 0);
  assert.strictEqual(utterances.length, 0);
});

test('energy VAD fires onSpeechStart once loud audio begins, onUtterance after enough trailing silence', () => {
  const starts = [];
  const utterances = [];
  const vad = createEnergyVad({ onSpeechStart: () => starts.push(1), onUtterance: (s) => utterances.push(s) });

  for (let i = 0; i < 3; i++) vad.feed(tone(100, 0.5)); // 300ms loud
  assert.strictEqual(starts.length, 1, 'should fire exactly once, not per loud chunk');
  assert.strictEqual(utterances.length, 0, 'not finalized yet -- no silence gap seen');

  for (let i = 0; i < 8; i++) vad.feed(tone(100, 0.001)); // 800ms fed; finalize fires the instant 700ms is crossed
  assert.strictEqual(utterances.length, 1);
  // 300ms loud + exactly the 700ms silence threshold (finalize fires the instant it's crossed,
  // so the 8th fed chunk arrives after reset and is dropped, not buffered).
  assert.strictEqual(utterances[0].length, tone(300, 0).length + tone(700, 0).length, 'buffer includes the trailing silence up to the threshold, not just the loud part');
});

test('energy VAD does not finalize on a brief pause shorter than the silence gap', () => {
  const utterances = [];
  const vad = createEnergyVad({ onSpeechStart: () => {}, onUtterance: (s) => utterances.push(s) });
  vad.feed(tone(200, 0.5));
  vad.feed(tone(200, 0.001)); // 200ms pause -- well under the 700ms gap
  vad.feed(tone(200, 0.5)); // resumes as the SAME utterance
  assert.strictEqual(utterances.length, 0);
  vad.flush();
  assert.strictEqual(utterances.length, 1);
  assert.ok(utterances[0].length > tone(500, 0).length, 'both speech bursts must be in one utterance');
});

test('energy VAD forces a finalize past the max-utterance cap during continuous speech', () => {
  const utterances = [];
  const vad = createEnergyVad({ onSpeechStart: () => {}, onUtterance: (s) => utterances.push(s) });
  for (let i = 0; i < 305; i++) vad.feed(tone(100, 0.5)); // 30.5s continuous, no pause
  assert.strictEqual(utterances.length, 1, 'must not grow one buffer unboundedly on nonstop speech');
});

test('energy VAD flush() is a no-op when nothing is buffered', () => {
  const utterances = [];
  const vad = createEnergyVad({ onSpeechStart: () => {}, onUtterance: (s) => utterances.push(s) });
  vad.flush();
  assert.strictEqual(utterances.length, 0);
});

test('energy VAD reports isSpeaking correctly across an utterance lifecycle', () => {
  const vad = createEnergyVad({ onSpeechStart: () => {}, onUtterance: () => {} });
  assert.strictEqual(vad.isSpeaking, false);
  vad.feed(tone(100, 0.5));
  assert.strictEqual(vad.isSpeaking, true);
  vad.flush();
  assert.strictEqual(vad.isSpeaking, false);
});

// --- whisper hallucination scrub ------------------------------------------------

test('scrubHallucination drops common Whisper silence hallucinations', () => {
  assert.strictEqual(scrubHallucination('Thank you.'), '');
  assert.strictEqual(scrubHallucination('Thanks for watching.'), '');
  assert.strictEqual(scrubHallucination('Clear conversation.'), '');
  assert.strictEqual(scrubHallucination('uh'), '');
});

test('scrubHallucination keeps real meeting speech', () => {
  assert.strictEqual(
    scrubHallucination('We will ship the build on Friday'),
    'We will ship the build on Friday'
  );
});
