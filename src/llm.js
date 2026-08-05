// Loads RunAnywhere's engine (Vulkan-accelerated LLM) and answers questions
// grounded in the rolling transcript.
const ELECTRON_SDK_DIST =
  process.env.RCLI_MEET_SDK_DIST ||
  'D:/the_code/runanywhere/SDK/runanywhere-sdks-main/sdk/runanywhere-electron/dist';

// Any RunAnywhere LLM catalog id (e.g. "qwen2.5-3b", auto-downloaded) or a
// local GGUF path works here; override with RCLI_MEET_LLM_PATH.
const DEFAULT_LLM_PATH = process.env.RCLI_MEET_LLM_PATH || 'qwen2.5-3b';
const DEFAULT_EMBEDDER_ID = process.env.RCLI_MEET_EMBEDDER_ID || 'minilm';

// Reasoning models (Qwen3 et al.) spend tokens inside <think>...</think>
// before answering. At 200 the whole budget got consumed thinking and the
// actual answer never arrived, so give it real headroom -- while still
// leaving room for the transcript: CONTEXT + ANSWER must stay under n_ctx.
const ANSWER_MAX_TOKENS = Number(process.env.RCLI_MEET_ANSWER_TOKENS) || 400;
const ANSWER_TEMPERATURE = 0.3;

// Better than paying for reasoning and hiding it: suppress it at the source.
// commons has a disable_thinking option, but the Electron bindings never
// expose it (GenerateOptions has no such field), so apply Qwen3's
// prompt-level soft switch -- the same approach the SDK's own
// Playground/android-use-agent uses, where it cut output from 381 tokens to
// 19-24. Set RCLI_MEET_THINKING=on to keep chain-of-thought enabled.
const NO_THINK_DIRECTIVE = '/no_think';
const THINKING_FORCED_ON = /^(1|on|true|yes)$/i.test(process.env.RCLI_MEET_THINKING || '');

/**
 * Whether this model honors the /no_think directive. It's a Qwen3 chat-template
 * feature; R1-style distillations reason because of the distillation itself and
 * ignore it (see the SDK's own ASSESSMENT.md), which is why the streaming
 * <think> filter stays in place as a safety net either way.
 */
function supportsNoThink(modelId) {
  return /qwen/i.test(String(modelId || ''));
}

// The transcript grows without bound, but the model's context does not: the
// llama.cpp backend computes `available = n_ctx - prompt_tokens - reserved`
// and (a) silently clamps max_tokens to it, then (b) hard-fails "Prompt too
// long" once it goes non-positive. With the common n_ctx=2048 fit, an
// unbounded 20-minute transcript blows that out after ~10 minutes of speech.
// So cap what we put in the prompt, keeping the MOST RECENT lines.
// ~3.5 chars/token is a conservative estimate for English ASR output.
const CHARS_PER_TOKEN = 3.5;
const DEFAULT_CONTEXT_TOKENS = 1200; // leaves room for answer + scaffolding at n_ctx=2048
const CONTEXT_TOKEN_BUDGET = Number(process.env.RCLI_MEET_CONTEXT_TOKENS) || DEFAULT_CONTEXT_TOKENS;
const CONTEXT_CHAR_BUDGET = Math.floor(CONTEXT_TOKEN_BUDGET * CHARS_PER_TOKEN);
// Split the budget: retrieved "earlier moments" are usually a few short lines,
// the recency window gets the rest.
const RETRIEVED_CHAR_SHARE = 0.35;

/**
 * Keep whole lines from the END of `lines` (most recent) that fit in `budget`
 * characters. Returns {text, dropped} so callers can be honest about
 * truncation instead of silently losing transcript.
 */
function fitLinesFromEnd(lines, budget) {
  const kept = [];
  let used = 0;
  for (let i = lines.length - 1; i >= 0; i--) {
    const cost = lines[i].length + 1; // +1 for the newline
    if (used + cost > budget) break;
    kept.unshift(lines[i]);
    used += cost;
  }
  return { text: kept.join('\n'), dropped: lines.length - kept.length };
}

async function loadEngine({ llmPath = DEFAULT_LLM_PATH, embedderId = DEFAULT_EMBEDDER_ID } = {}) {
  // Required lazily so a bad RCLI_MEET_SDK_DIST surfaces as a clear message
  // through the caller's error handling rather than a raw MODULE_NOT_FOUND
  // stack at require time.
  let RunAnywhere;
  try {
    ({ RunAnywhere } = require(ELECTRON_SDK_DIST));
  } catch (err) {
    throw new Error(
      `could not load the RunAnywhere Electron SDK from:\n  ${ELECTRON_SDK_DIST}\n` +
        `Set RCLI_MEET_SDK_DIST to your @runanywhere/electron "dist" directory.\n` +
        `(underlying error: ${err.message})`
    );
  }

  RunAnywhere.initialize({ environment: 'development' });

  let llm;
  let embedder;
  try {
    llm = await RunAnywhere.loadLLM(llmPath);
    embedder = await RunAnywhere.loadEmbedder(embedderId);
  } catch (err) {
    // Don't leak a half-initialized engine (and a loaded multi-GB model) if
    // only the second load failed.
    try {
      if (llm) llm.unload();
      RunAnywhere.shutdown();
    } catch {
      /* best-effort cleanup; report the original failure below */
    }
    throw err;
  }

  const disableThinking = !THINKING_FORCED_ON && supportsNoThink(llmPath);

  let shutDown = false;
  return {
    llm,
    embedder,
    disableThinking,
    shutdown() {
      if (shutDown) return;
      shutDown = true;
      try {
        llm.unload();
        embedder.unload();
      } finally {
        RunAnywhere.shutdown();
      }
    },
  };
}

/**
 * @param recentLines {string[]} caption lines in the recency window, oldest first
 * @param retrievedLines {string[]} similarity-matched lines from earlier in the session
 * @param partial {string} the in-flight, not-yet-finalized utterance (may be '')
 * @param question {string}
 */
function buildPrompt({
  recentLines = [],
  retrievedLines = [],
  partial = '',
  question,
  disableThinking = false,
}) {
  const retrievedFit = fitLinesFromEnd(retrievedLines, CONTEXT_CHAR_BUDGET * RETRIEVED_CHAR_SHARE);
  // Whatever the retrieved section didn't use goes to the recency window.
  const recentBudget = CONTEXT_CHAR_BUDGET - retrievedFit.text.length;
  const partialLine = partial ? `[now] ${partial}` : '';
  // The in-flight utterance is the most likely thing a question is about, so
  // it gets reserved space ahead of older finalized lines.
  const recentFit = fitLinesFromEnd(recentLines, recentBudget - partialLine.length);

  const recentText = [recentFit.text, partialLine].filter(Boolean).join('\n');
  const truncationNote =
    recentFit.dropped > 0
      ? `\n(note: ${recentFit.dropped} earlier line(s) omitted to fit the context window)`
      : '';

  return `You are answering questions about a live meeting/talk transcript. Use ONLY the transcript excerpts below. If the answer isn't in them, say so briefly. Be concise.

Relevant earlier moments:
${retrievedFit.text || '(none)'}

Recent transcript:${truncationNote}
${recentText || '(none yet)'}

Question: ${question}
Answer:${disableThinking ? `\n${NO_THINK_DIRECTIVE}` : ''}`;
}

const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';
// Whitespace-only <think> blocks (what Qwen3 emits under /no_think) are not
// real reasoning and shouldn't trigger a user-visible indicator.
const EMPTY_THINK_TOLERANCE = 8;

/** Length of the longest suffix of `s` that is a proper prefix of `tag`. */
function partialTagTail(s, tag) {
  const max = Math.min(s.length, tag.length - 1);
  for (let n = max; n > 0; n--) {
    if (s.endsWith(tag.slice(0, n))) return n;
  }
  return 0;
}

/**
 * Everything outside <think>...</think>. An unterminated <think> hides the
 * rest (we're mid-reasoning), and a partial tag at the very end is held back
 * so a half-written "<thin" never reaches the terminal.
 */
function visibleOutside(raw) {
  let out = '';
  let i = 0;
  let thinking = false;
  let thinkingChars = 0;
  while (i < raw.length) {
    const open = raw.indexOf(THINK_OPEN, i);
    if (open === -1) {
      out += raw.slice(i);
      break;
    }
    out += raw.slice(i, open);
    const close = raw.indexOf(THINK_CLOSE, open + THINK_OPEN.length);
    if (close === -1) {
      // Still inside the block; hide everything after it.
      thinking = true;
      thinkingChars += raw.length - (open + THINK_OPEN.length);
      break;
    }
    thinkingChars += close - (open + THINK_OPEN.length);
    i = close + THINK_CLOSE.length;
  }
  const hold = partialTagTail(out, THINK_OPEN);
  if (hold) out = out.slice(0, out.length - hold);
  return { text: out, thinking, thinkingChars };
}

/**
 * Incremental version of visibleOutside for a token stream: push() returns
 * only the newly-revealed visible text (possibly '').
 */
function createThinkFilter() {
  let raw = '';
  let emitted = 0;
  return {
    push(token) {
      raw += token;
      const { text, thinking, thinkingChars } = visibleOutside(raw);
      let delta = '';
      if (text.length > emitted) {
        delta = text.slice(emitted);
        emitted = text.length;
      }
      return { delta, thinking, thinkingChars };
    },
    get visibleText() {
      return visibleOutside(raw).text;
    },
    get rawText() {
      return raw;
    },
  };
}

/**
 * Stream an answer with reasoning-block filtering.
 * @param onToken {(text: string) => void} visible answer text only
 * @param onThinking {() => void} called once if the model starts reasoning
 * @returns {{answer: string, raw: string}}
 */
async function askQuestion(llm, promptCtx, onToken, onThinking = () => {}) {
  const prompt = buildPrompt(promptCtx);
  const filter = createThinkFilter();
  let announcedThinking = false;

  for await (const token of llm.generate(prompt, {
    maxTokens: ANSWER_MAX_TOKENS,
    temperature: ANSWER_TEMPERATURE,
  })) {
    const { delta, thinking, thinkingChars } = filter.push(token);
    // Qwen3 under /no_think still emits an EMPTY <think></think> block, so
    // don't flash a "thinking" indicator unless it's actually reasoning.
    if (thinking && thinkingChars > EMPTY_THINK_TOLERANCE && !announcedThinking) {
      announcedThinking = true;
      onThinking();
    }
    if (delta) onToken(delta);
  }

  return { answer: filter.visibleText.trim(), raw: filter.rawText };
}

module.exports = {
  loadEngine,
  buildPrompt,
  askQuestion,
  fitLinesFromEnd,
  createThinkFilter,
  visibleOutside,
  supportsNoThink,
  NO_THINK_DIRECTIVE,
  DEFAULT_LLM_PATH,
  DEFAULT_EMBEDDER_ID,
  CONTEXT_TOKEN_BUDGET,
  ANSWER_MAX_TOKENS,
};
