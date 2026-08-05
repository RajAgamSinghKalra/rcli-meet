#!/usr/bin/env node
// rcli-meet: offline live-meeting captions + local-LLM Q&A over the rolling
// transcript. CLI only, no UI. Everything runs locally: RunAnywhere's
// Vulkan-accelerated LLM/embedder engine + the official sherpa-onnx streaming
// recognizer + a WASAPI loopback capture helper. Nothing leaves the machine.
const path = require('path');
const readline = require('readline');

const { startCapture } = require('./capture');
const { createStreamingSTT } = require('./stt');
const { createTranscript } = require('./transcript');
const { createRetrieval } = require('./retrieval');
const { loadEngine, buildPrompt, DEFAULT_LLM_PATH, DEFAULT_EMBEDDER_ID } = require('./llm');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'sherpa-onnx-streaming-zipformer-en-2023-06-26');
const TRANSCRIPTS_DIR = path.join(__dirname, '..', 'transcripts');
const RETRIEVAL_TOP_K = 5;

function parseArgs(argv) {
  const opts = { minutes: 20, llmPath: DEFAULT_LLM_PATH, embedderId: DEFAULT_EMBEDDER_ID };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--minutes') opts.minutes = Number(argv[++i]);
    else if (argv[i] === '--llm') opts.llmPath = argv[++i];
    else if (argv[i] === '--embedder') opts.embedderId = argv[++i];
  }
  return opts;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  console.log('[rcli-meet] loading local engine (LLM + embedder, Vulkan)...');
  const engine = await loadEngine({ llmPath: opts.llmPath, embedderId: opts.embedderId });
  console.log(`[rcli-meet] LLM ready: ${opts.llmPath}`);
  console.log(`[rcli-meet] embedder ready: ${opts.embedderId}`);

  console.log('[rcli-meet] loading streaming STT model...');
  const stt = createStreamingSTT(MODEL_DIR);
  console.log('[rcli-meet] STT ready.');

  const transcript = createTranscript(TRANSCRIPTS_DIR);
  console.log(`[rcli-meet] session log: ${transcript.logPath}`);
  const retrieval = createRetrieval(engine.embedder);

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });

  function printLine(line) {
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write(line + '\n');
    rl.prompt(true);
  }

  let currentPartial = '';

  stt.on('partial', (text) => {
    currentPartial = text;
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write(`… ${text}`);
  });

  stt.on('final', (text) => {
    currentPartial = '';
    const seg = transcript.add(text);
    retrieval.add(seg);
    printLine(seg.line);
  });

  const capture = startCapture((samples) => stt.feed(samples));

  let answering = false;
  rl.on('line', async (line) => {
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
    // Include the in-flight (not-yet-finalized) utterance too -- endpoint
    // detection needs a trailing silence gap, so whatever was *just* said is
    // often still "partial" when a question comes in about it.
    const windowText = [transcript.windowText(opts.minutes), currentPartial ? `[now] ${currentPartial}` : '']
      .filter(Boolean)
      .join('\n');
    const retrieved = retrieval.topK(question, RETRIEVAL_TOP_K);
    const promptCtx = { windowText, retrieved, question };

    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write('>> ');
    try {
      for await (const token of engine.llm.generate(buildPrompt(promptCtx), {
        maxTokens: 200,
        temperature: 0.3,
      })) {
        process.stdout.write(token);
      }
    } catch (err) {
      process.stdout.write(`\n[rcli-meet] answer failed: ${err.message}`);
    }
    process.stdout.write('\n');
    answering = false;
    rl.prompt(true);
  });

  let shuttingDown = false;
  async function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log('\n[rcli-meet] shutting down...');
    capture.stop();
    stt.close();
    transcript.close();
    engine.shutdown();
    rl.close();
    process.exit(0);
  }

  rl.on('SIGINT', shutdown);
  process.on('SIGINT', shutdown);

  console.log('[rcli-meet] listening. Type a question and press Enter, or /quit to exit.\n');
  rl.prompt();
}

main().catch((err) => {
  // stdout, not stderr/console.error -- run.bat redirects stderr to NUL to
  // silence the native addon's log spam, so real errors must go to stdout
  // to stay visible.
  console.log('[rcli-meet] fatal error:', err);
  process.exit(1);
});
