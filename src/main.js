#!/usr/bin/env node
// rcli-meet: offline live-meeting captions + local-LLM Q&A over the rolling
// transcript. CLI only, no UI. Everything runs locally: RunAnywhere's
// Vulkan-accelerated LLM/embedder engine + the official sherpa-onnx streaming
// recognizer + a WASAPI loopback capture helper. Nothing leaves the machine.
const path = require('path');
const readline = require('readline');

const { startCapture } = require('./capture');
const { createStreamingSTT, assertModelPresent } = require('./stt');
const { createTranscript } = require('./transcript');
const { createRetrieval } = require('./retrieval');
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

const USAGE = `rcli-meet -- offline live captions + local-LLM Q&A over the rolling transcript

Usage: node src/main.js [options]

Options:
  --minutes <n>        Size of the recency window fed to the model (default: 20)
  --llm <id|path>      RunAnywhere LLM catalog id or local GGUF path
  --embedder <id>      RunAnywhere embedder catalog id
  -h, --help           Show this help

Environment:
  RCLI_MEET_SDK_DIST       Path to your @runanywhere/electron "dist" directory
  RCLI_MEET_LLM_PATH       Default LLM (catalog id or GGUF path)
  RCLI_MEET_EMBEDDER_ID    Default embedder catalog id
  RCLI_MEET_PYTHON         Python interpreter for the loopback capture helper
  RCLI_MEET_STT_MODEL_DIR  Streaming Zipformer model directory
  RCLI_MEET_CONTEXT_TOKENS Transcript tokens to fit in the prompt (default: 1200)

In-session: type a question and press Enter; /quit or Ctrl+C to exit.`;

class UsageError extends Error {}

function parseArgs(argv) {
  const opts = { minutes: 20, llmPath: DEFAULT_LLM_PATH, embedderId: DEFAULT_EMBEDDER_ID };

  // Every flag below needs a value; bail out clearly rather than silently
  // consuming the next flag (or undefined) as one.
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
  console.log(`[rcli-meet] LLM ready: ${opts.llmPath}`);
  console.log(`[rcli-meet] embedder ready: ${opts.embedderId}`);

  console.log('[rcli-meet] loading streaming STT model...');
  const stt = createStreamingSTT(MODEL_DIR);
  console.log('[rcli-meet] STT ready.');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
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

  let currentPartial = '';

  stt.on('partial', (text) => {
    currentPartial = text;
    if (answering) return; // tracked for context, just not repainted
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write(`… ${text}`);
  });

  stt.on('final', (text) => {
    currentPartial = '';
    // Always record, even mid-answer -- only the rendering is deferred.
    const seg = transcript.add(text);
    retrieval.add(seg);
    if (answering) deferredLines.push(seg.line);
    else printLine(seg.line);
  });

  let fatal = null;
  const capture = startCapture(
    (samples) => {
      // A throw here would escape through the child-process 'data' event as an
      // uncaught exception and kill the session.
      try {
        stt.feed(samples);
      } catch (err) {
        notify(`transcription error: ${err.message}`);
      }
    },
    (message) => {
      fatal = message;
      notify(message);
      notify('captions have stopped; you can still ask about what was already transcribed.');
    }
  );

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

      // Reasoning models emit a <think> block first. Show a placeholder so the
      // wait isn't silent, then replace it with the real answer.
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
            recentLines,
            retrievedLines: retrieved.map((r) => r.line),
            partial: currentPartial,
            question,
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
    capture.stop();
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
