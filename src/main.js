#!/usr/bin/env node
// rcli-meet: offline live-meeting captions + local-LLM Q&A over the rolling
// transcript. CLI only, no UI. Everything runs locally: RunAnywhere's
// Vulkan-accelerated LLM/embedder engine + the official sherpa-onnx streaming
// recognizer + WASAPI loopback (the meeting) and mic (you) capture helpers.
// Nothing leaves the machine.
const path = require('path');
const readline = require('readline');

const { startCapture } = require('./capture');
const { createSTTEngine, assertModelPresent } = require('./stt');
const { createTranscript } = require('./transcript');
const { createRetrieval } = require('./retrieval');
const { createSummarizer } = require('./summary');
const {
  loadEngine,
  askQuestion,
  DEFAULT_LLM_PATH,
  DEFAULT_EMBEDDER_ID,
  CONTEXT_TOKEN_BUDGET,
} = require('./llm');

const MODEL_DIR =
  process.env.RCLI_MEET_STT_MODEL_DIR ||
  path.join(__dirname, '..', 'models', 'sherpa-onnx-streaming-zipformer-en-2023-06-26');
const TRANSCRIPTS_DIR = path.join(__dirname, '..', 'transcripts');
const RETRIEVAL_TOP_K = 5;
const SOURCES = ['meeting', 'you'];

const USAGE = `rcli-meet -- offline live captions + local-LLM Q&A over the rolling transcript

Usage: node src/main.js [options]

Options:
  --minutes <n>        Size of the recency window fed to the model (default: 20)
  --llm <id|path>      RunAnywhere LLM catalog id or local GGUF path
  --embedder <id>      RunAnywhere embedder catalog id
  --no-mic             Don't capture your own microphone, meeting audio only
  -h, --help           Show this help

Environment:
  RCLI_MEET_SDK_DIST       Path to your @runanywhere/electron "dist" directory
  RCLI_MEET_LLM_PATH       Default LLM (catalog id or GGUF path)
  RCLI_MEET_EMBEDDER_ID    Default embedder catalog id
  RCLI_MEET_PYTHON         Python interpreter for the audio capture helper
  RCLI_MEET_STT_MODEL_DIR  Streaming Zipformer model directory
  RCLI_MEET_CONTEXT_TOKENS Transcript tokens to fit in the prompt (default: 1200)
  RCLI_MEET_SUMMARY_EVERY  Finalized segments between summary updates (default: 8)
  RCLI_MEET_SUMMARY_TOKENS Token budget for each summary update (default: 220)

In-session: type a question and press Enter; /quit or Ctrl+C to exit.

Note: without headphones, your microphone may also pick up the meeting audio
itself (speaker bleed), which would mislabel meeting speech as "you" said it.`;

class UsageError extends Error {}

function parseArgs(argv) {
  const opts = { minutes: 20, llmPath: DEFAULT_LLM_PATH, embedderId: DEFAULT_EMBEDDER_ID, mic: true };

  const takeValue = (flag, i) => {
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new UsageError(`${flag} needs a value`);
    }
    return value;
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') {
      opts.help = true;
    } else if (arg === '--minutes') {
      const raw = takeValue(arg, i++);
      const minutes = Number(raw);
      // Number('abc') is NaN, which made every window comparison false and
      // silently produced an empty transcript context.
      if (!Number.isFinite(minutes) || minutes <= 0) {
        throw new UsageError(`--minutes must be a positive number (got "${raw}")`);
      }
      opts.minutes = minutes;
    } else if (arg === '--llm') {
      opts.llmPath = takeValue(arg, i++);
    } else if (arg === '--embedder') {
      opts.embedderId = takeValue(arg, i++);
    } else if (arg === '--no-mic') {
      opts.mic = false;
    } else {
      throw new UsageError(`unknown option "${arg}"`);
    }
  }
  return opts;
}

async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    if (err instanceof UsageError) {
      console.log(`[rcli-meet] ${err.message}\n\n${USAGE}`);
      process.exit(2);
    }
    throw err;
  }
  if (opts.help) {
    console.log(USAGE);
    return;
  }

  // Validate the STT model before spending ~15s loading a multi-GB LLM only to
  // die on a missing .onnx afterwards.
  assertModelPresent(MODEL_DIR);

  console.log('[rcli-meet] loading local engine (LLM + embedder, Vulkan)...');
  const engine = await loadEngine({ llmPath: opts.llmPath, embedderId: opts.embedderId });
  console.log(
    `[rcli-meet] LLM ready: ${opts.llmPath}` +
      (engine.disableThinking ? ' (chain-of-thought suppressed)' : '')
  );
  console.log(`[rcli-meet] embedder ready: ${opts.embedderId}`);

  console.log('[rcli-meet] loading streaming STT model...');
  const stt = createSTTEngine(MODEL_DIR);
  console.log('[rcli-meet] STT ready.');
  if (opts.mic) {
    console.log(
      '[rcli-meet] capturing meeting audio + your mic. Without headphones, your mic may ' +
        'also pick up the meeting itself (speaker bleed) and mislabel it as "you".'
    );
  } else {
    console.log('[rcli-meet] capturing meeting audio only (--no-mic).');
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
    // Key interactivity off stdin, not stdout: when launched via quiet.js our
    // stdout is a pipe (so readline would default terminal:false and stop
    // echoing), but stdin is still the real console and the escape sequences
    // pass through the filter to it unchanged.
    terminal: !!process.stdin.isTTY,
  });

  let answering = false;
  // Captions that arrive mid-answer. Repainting the caption line while an
  // answer is streaming issues clearLine/cursorTo escapes that wipe out the
  // partially-written answer, visibly shredding both. Hold them and flush
  // once the answer is done.
  const deferredLines = [];

  function printLine(line) {
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write(line + '\n');
    rl.prompt(true);
  }

  function notify(message) {
    const line = `[rcli-meet] ${message}`;
    if (answering) deferredLines.push(line);
    else printLine(line);
  }

  const transcript = createTranscript(TRANSCRIPTS_DIR, { onError: notify });
  console.log(`[rcli-meet] session log: ${transcript.logPath}`);
  const retrieval = createRetrieval(engine.embedder, { onError: notify });
  const summarizer = createSummarizer({
    llm: engine.llm,
    disableThinking: engine.disableThinking,
    onError: notify,
  });

  // One in-flight (not-yet-finalized) utterance per source -- endpoint
  // detection needs a trailing silence gap, so whatever was *just* said is
  // often still "partial" when a question comes in about it.
  const partials = { meeting: '', you: '' };
  // When each source's CURRENT (not yet finalized) utterance began, so a
  // finalized segment can carry a real [start, end] range -- needed to
  // detect genuine cross-talk (both sources active at once) rather than just
  // sequential turns. Reset to null on finalize.
  const utteranceStart = { meeting: null, you: null };

  /**
   * clearLine(0) + cursorTo(0) only reset the CURRENT terminal row. If the
   * previous write wrapped onto a second row (long caption, narrow window),
   * those escapes don't touch that leftover row -- each update then stacks a
   * new copy below the last instead of replacing it. Truncating to fit one
   * row (keeping the tail -- most recent words matter most) makes wrapping
   * impossible, which is a simpler fix than tracking multi-row cursor state.
   */
  function fitOneRow(text) {
    const width = (process.stdout.columns || 80) - 1;
    return text.length <= width ? text : '…' + text.slice(-(width - 1));
  }

  /** Wires partial/final handling for one source's STT stream. */
  function wireStream(sttStream, source) {
    sttStream.on('partial', (text) => {
      if (utteranceStart[source] == null) utteranceStart[source] = transcript.elapsedNow();
      partials[source] = text;
      if (answering) return; // tracked for context, just not repainted
      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      process.stdout.write(fitOneRow(`… [${source}] ${text}`));
    });

    sttStream.on('final', (text) => {
      const startedAt = utteranceStart[source] ?? transcript.elapsedNow();
      utteranceStart[source] = null;
      partials[source] = '';
      // Always record, even mid-answer -- only the rendering is deferred.
      const seg = transcript.add(text, source, startedAt);
      retrieval.add(seg);
      summarizer.addSegment(seg);
      summarizer.maybeUpdate();
      if (answering) deferredLines.push(seg.line);
      else printLine(seg.line);
    });
  }

  const sttStreams = {};
  const captures = {};
  for (const source of SOURCES) {
    if (source === 'you' && !opts.mic) continue;
    const sttStream = stt.createStream();
    wireStream(sttStream, source);
    sttStreams[source] = sttStream;

    captures[source] = startCapture(
      source === 'meeting' ? 'loopback' : 'mic',
      (samples) => {
        // A throw here would escape through the child-process 'data' event as
        // an uncaught exception and kill the session.
        try {
          sttStream.feed(samples);
        } catch (err) {
          notify(`${source} transcription error: ${err.message}`);
        }
      },
      (message) => {
        notify(message);
        notify(`${source} captions have stopped; other sources keep working.`);
      }
    );
  }

  rl.on('line', (line) => {
    void handleLine(line);
  });

  async function handleLine(line) {
    const question = line.trim();
    if (!question) {
      rl.prompt();
      return;
    }
    if (question === '/quit' || question === '/exit') {
      await shutdown();
      return;
    }
    if (answering) {
      printLine('[rcli-meet] still answering the previous question, please wait...');
      return;
    }

    answering = true;
    try {
      // The recency window and the similarity hits are drawn from the same
      // transcript, so exclude lines already in the window from "earlier
      // moments" -- otherwise the prompt repeats itself and wastes the
      // context budget.
      const recentSegments = transcript.lastMinutes(opts.minutes);
      const recentLines = recentSegments.map((s) => s.line);
      const inWindow = new Set(recentLines);
      const retrieved = retrieval.topK(question, RETRIEVAL_TOP_K, inWindow);

      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      process.stdout.write('>> ');
      let placeholderShown = false;
      let gotAnswerText = false;
      const clearPlaceholder = () => {
        if (!placeholderShown) return;
        readline.clearLine(process.stdout, 0);
        readline.cursorTo(process.stdout, 0);
        process.stdout.write('>> ');
        placeholderShown = false;
      };

      try {
        const { answer } = await askQuestion(
          engine.llm,
          {
            summary: summarizer.summary,
            recentLines,
            retrievedLines: retrieved.map((r) => r.line),
            partials,
            question,
            disableThinking: engine.disableThinking,
          },
          (text) => {
            clearPlaceholder();
            gotAnswerText = true;
            process.stdout.write(text);
          },
          () => {
            process.stdout.write('(thinking...)');
            placeholderShown = true;
          }
        );
        clearPlaceholder();
        if (!gotAnswerText || !answer) {
          process.stdout.write(
            '[rcli-meet] the model used its whole budget reasoning without answering. ' +
              'Try a more specific question, or raise RCLI_MEET_ANSWER_TOKENS.'
          );
        }
      } catch (err) {
        clearPlaceholder();
        process.stdout.write(`\n[rcli-meet] answer failed: ${err.message}`);
      }
      process.stdout.write('\n');
    } finally {
      answering = false;
    }

    while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
    rl.prompt(true);
  }

  let shuttingDown = false;
  async function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;
    process.stdout.write('\n[rcli-meet] shutting down...\n');
    for (const source of Object.keys(captures)) captures[source].stop();
    for (const source of Object.keys(sttStreams)) sttStreams[source].close();
    stt.close();
    // Must await: logStream.end() is async, and exiting first truncates the
    // tail of the session log.
    await transcript.close();
    engine.shutdown();
    rl.close();
    process.exit(0);
  }

  rl.on('SIGINT', () => void shutdown());
  process.on('SIGINT', () => void shutdown());

  console.log(
    `[rcli-meet] listening (window: ${opts.minutes} min, context budget: ${CONTEXT_TOKEN_BUDGET} tokens).`
  );
  console.log('[rcli-meet] Type a question and press Enter, or /quit to exit.\n');
  rl.prompt();
}

main().catch((err) => {
  // stdout, not stderr/console.error -- run.bat redirects stderr to NUL to
  // silence the native addon's log spam, so real errors must go to stdout
  // to stay visible.
  console.log(`[rcli-meet] fatal error: ${err && err.message ? err.message : err}`);
  process.exit(1);
});
