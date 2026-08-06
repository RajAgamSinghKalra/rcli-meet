#!/usr/bin/env node
// rcli-meet: offline live-meeting captions + local-LLM Q&A over the rolling
// transcript, with voice commands, TTS replies, and save/load. CLI only, no
// UI. Everything runs locally. Nothing leaves the machine.
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const { startCapture } = require('./capture');
// Whisper (sttWhisper.js) is the default: the streaming Zipformer model
// (stt.js) is small, English-only, and trained mostly on native-accent
// speech -- it does badly on non-native accents (Indian English among
// them). Whisper was trained on far more diverse, heavily-accented
// multilingual audio and is meaningfully more accurate here, at the cost of
// true word-by-word streaming (results arrive per-utterance, on a silence
// gap, not live per-word). Set RCLI_MEET_STT_ENGINE=zipformer for the old
// low-latency streaming behavior instead.
const STT_ENGINE = (process.env.RCLI_MEET_STT_ENGINE || 'whisper').toLowerCase();
const { createSTTEngine, assertModelPresent } = require(STT_ENGINE === 'zipformer' ? './stt' : './sttWhisper');
const { createTranscript } = require('./transcript');
const { createRetrieval } = require('./retrieval');
const { createSummarizer } = require('./summary');
const { createTTS } = require('./tts');
const { parseCommand } = require('./commands');
const { getActiveWindowTitle } = require('./activeWindow');
const { newSessionDir, saveSession, addFileToSession, listSessions, loadSession, DEFAULT_SESSIONS_DIR } =
  require('./session');
const {
  loadEngine,
  askQuestion,
  DEFAULT_LLM_PATH,
  DEFAULT_EMBEDDER_ID,
  CONTEXT_TOKEN_BUDGET,
} = require('./llm');

const DEFAULT_MODEL_DIR =
  STT_ENGINE === 'zipformer'
    ? path.join(__dirname, '..', 'models', 'sherpa-onnx-streaming-zipformer-en-2023-06-26')
    : path.join(__dirname, '..', 'models', 'sherpa-onnx-whisper-small.en');
const MODEL_DIR = process.env.RCLI_MEET_STT_MODEL_DIR || DEFAULT_MODEL_DIR;
const RETRIEVAL_TOP_K = 5;
const SOURCES = ['meeting', 'you'];
const FILE_CHUNK_CHARS = 1500;

const USAGE = `rcli-meet -- offline live captions + local-LLM Q&A over the rolling transcript

Usage: node src/main.js [options]

Options:
  --minutes <n>        Size of the recency window fed to the model (default: 20)
  --llm <id|path>       RunAnywhere LLM catalog id or local GGUF path
  --embedder <id>      RunAnywhere embedder catalog id
  --no-mic             Don't capture your own microphone, meeting audio only
  --no-tts             Don't speak answers out loud
  -h, --help           Show this help

In-session (typed OR spoken, except stop which is typed-only):
  start / record       Begin a new recording session (transcribes + stores both sources)
  stop                 (typed only) Stop the current recording session
  save                 Save the current session to sessions/<date>_<app>/
  load                 List saved sessions and pick one to load into context
  add <path>           Ingest a .txt file into the current session (typed only)
  anything else        Ask a question -- answered from the transcript, always spoken back
  /quit                Exit

Environment: see README.md (RCLI_MEET_SDK_DIST, RCLI_MEET_LLM_PATH, RCLI_MEET_PYTHON, etc.)

Note: without headphones, your microphone may also pick up the meeting audio
itself (speaker bleed), which would mislabel meeting speech as "you" said it.`;

class UsageError extends Error {}

function parseArgs(argv) {
  const opts = { minutes: 20, llmPath: DEFAULT_LLM_PATH, embedderId: DEFAULT_EMBEDDER_ID, mic: true, tts: true };
  const takeValue = (flag, i) => {
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) throw new UsageError(`${flag} needs a value`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') opts.help = true;
    else if (arg === '--minutes') {
      const raw = takeValue(arg, i++);
      const minutes = Number(raw);
      if (!Number.isFinite(minutes) || minutes <= 0) throw new UsageError(`--minutes must be a positive number (got "${raw}")`);
      opts.minutes = minutes;
    } else if (arg === '--llm') opts.llmPath = takeValue(arg, i++);
    else if (arg === '--embedder') opts.embedderId = takeValue(arg, i++);
    else if (arg === '--no-mic') opts.mic = false;
    else if (arg === '--no-tts') opts.tts = false;
    else throw new UsageError(`unknown option "${arg}"`);
  }
  return opts;
}

/** Splits text on paragraph/sentence boundaries into ~FILE_CHUNK_CHARS chunks. */
function chunkText(text, maxChars = FILE_CHUNK_CHARS) {
  const paragraphs = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  const chunks = [];
  let current = '';
  for (const p of paragraphs) {
    if (current && current.length + p.length + 2 > maxChars) {
      chunks.push(current);
      current = '';
    }
    current = current ? `${current}\n\n${p}` : p;
    while (current.length > maxChars) {
      chunks.push(current.slice(0, maxChars));
      current = current.slice(maxChars);
    }
  }
  if (current) chunks.push(current);
  return chunks.length ? chunks : [text.slice(0, maxChars)];
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

  assertModelPresent(MODEL_DIR);

  console.log('[rcli-meet] loading local engine (LLM + embedder, Vulkan)...');
  const engine = await loadEngine({ llmPath: opts.llmPath, embedderId: opts.embedderId });
  console.log(`[rcli-meet] LLM ready: ${opts.llmPath}` + (engine.disableThinking ? ' (chain-of-thought suppressed)' : ''));
  console.log(`[rcli-meet] embedder ready: ${opts.embedderId}`);

  console.log('[rcli-meet] loading streaming STT model...');
  const stt = createSTTEngine(MODEL_DIR);
  console.log(`[rcli-meet] STT ready (engine: ${STT_ENGINE}).`);

  const tts = opts.tts ? createTTS() : null;
  console.log(opts.tts ? '[rcli-meet] TTS ready -- answers will be spoken.' : '[rcli-meet] TTS disabled (--no-tts).');
  if (opts.mic) {
    console.log(
      '[rcli-meet] mic capture enabled. Without headphones, your mic may also pick up the meeting itself ' +
        '(speaker bleed) and mislabel it as "you".'
    );
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
    terminal: !!process.stdin.isTTY,
  });

  let answering = false;
  let speaking = false; // true while TTS plays; both STT streams are muted so it can't hear/transcribe itself
  let recording = false;
  let sessionDir = null; // survives stop(); replaced on the next start()
  let transcript = null;
  let retrieval = null;
  let summarizer = null;
  let awaitingLoadChoice = null; // string[] of session names, set right after "load" lists them
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

  function fitOneRow(text) {
    const width = (process.stdout.columns || 80) - 1;
    return text.length <= width ? text : '…' + text.slice(-(width - 1));
  }

  function ensureSessionState() {
    if (!transcript) transcript = createTranscript(sessionDir || DEFAULT_SESSIONS_DIR, { onError: notify });
    if (!retrieval) retrieval = createRetrieval(engine.embedder, { onError: notify });
    if (!summarizer) summarizer = createSummarizer({ llm: engine.llm, disableThinking: engine.disableThinking, onError: notify });
  }

  async function handleCommand(cmd) {
    if (cmd === 'start') {
      if (recording) return notify('already recording.');
      const appName = getActiveWindowTitle();
      sessionDir = newSessionDir(DEFAULT_SESSIONS_DIR, appName);
      fs.mkdirSync(sessionDir, { recursive: true });
      transcript = createTranscript(sessionDir, { onError: notify });
      retrieval = createRetrieval(engine.embedder, { onError: notify });
      summarizer = createSummarizer({ llm: engine.llm, disableThinking: engine.disableThinking, onError: notify });
      recording = true;
      notify(`recording started (window: "${appName}"). Session folder: ${sessionDir}`);
    } else if (cmd === 'stop') {
      if (!recording) return notify('not currently recording.');
      recording = false;
      notify('recording stopped. Ask questions by voice or text, or type "save".');
    } else if (cmd === 'save') {
      if (!sessionDir) return notify('nothing to save yet -- say or type "start" first.');
      ensureSessionState();
      const appName = path.basename(sessionDir).split('_').slice(1).join('_') || 'unknown-app';
      saveSession(sessionDir, { transcriptLogPath: transcript.logPath, summary: summarizer.summary, appName });
      notify(`saved to ${sessionDir}`);
    } else if (cmd === 'load') {
      if (recording) return notify('stop the current recording first (type "stop") before loading.');
      const names = listSessions(DEFAULT_SESSIONS_DIR);
      if (!names.length) return notify('no saved sessions found.');
      awaitingLoadChoice = names;
      notify('saved sessions:\n' + names.map((n, i) => `  ${i + 1}. ${n}`).join('\n') + '\nType a number to load, or anything else to cancel.');
    }
  }

  function resolveLoadChoice(input, names) {
    const idx = Number(input) - 1;
    const name = Number.isInteger(idx) && names[idx] ? names[idx] : names.find((n) => n === input.trim());
    if (!name) return notify('load cancelled.');
    const { dir, transcriptText, summary, files } = loadSession(DEFAULT_SESSIONS_DIR, name);
    sessionDir = dir;
    ensureSessionState();
    const lines = transcriptText.split('\n').map((l) => l.trim()).filter(Boolean);
    lines.forEach((line, i) => retrieval.add({ line, text: line, elapsedMs: i }));
    files.forEach((f) =>
      chunkText(f.text).forEach((chunk, i) =>
        retrieval.add({ line: `[file: ${f.name} #${i + 1}] ${chunk}`, text: chunk, elapsedMs: 0 })
      )
    );
    if (summary) {
      summarizer.setSummary(summarizer.summary ? `${summarizer.summary}\n\n(From a loaded session:)\n${summary}` : summary);
    }
    notify(`loaded "${name}" (${lines.length} transcript line(s), ${files.length} file(s)).`);
  }

  function handleAddFile(rawPath) {
    if (!sessionDir) return notify('start a session first ("start") before adding a file.');
    const filePath = rawPath.replace(/^["']|["']$/g, '');
    let text;
    try {
      addFileToSession(sessionDir, filePath);
      text = fs.readFileSync(filePath, 'utf8');
    } catch (err) {
      return notify(`could not add file: ${err.message}`);
    }
    ensureSessionState();
    const chunks = chunkText(text);
    const name = path.basename(filePath);
    chunks.forEach((chunk, i) => retrieval.add({ line: `[file: ${name} #${i + 1}] ${chunk}`, text: chunk, elapsedMs: 0 }));
    notify(`added "${name}" (${chunks.length} chunk(s)) -- it's now searchable for questions.`);
  }

  async function speakAnswer(text) {
    if (!tts || !text) return;
    speaking = true;
    try {
      await tts.speak(text);
    } catch (err) {
      notify(`TTS playback failed: ${err.message}`);
    } finally {
      speaking = false;
    }
  }

  async function handleQuestion(question) {
    if (answering) {
      printLine('[rcli-meet] still answering the previous question, please wait...');
      return;
    }
    answering = true;
    try {
      const recentLines = transcript ? transcript.lastMinutes(opts.minutes).map((s) => s.line) : [];
      const inWindow = new Set(recentLines);
      const retrieved = retrieval ? retrieval.topK(question, RETRIEVAL_TOP_K, inWindow) : [];
      const summary = summarizer ? summarizer.summary : '';

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

      let answer = '';
      try {
        ({ answer } = await askQuestion(
          engine.llm,
          {
            summary,
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
        ));
        clearPlaceholder();
        if (!gotAnswerText || !answer) {
          process.stdout.write(
            '[rcli-meet] the model used its whole budget reasoning without answering. Try a more specific question, or raise RCLI_MEET_ANSWER_TOKENS.'
          );
        }
      } catch (err) {
        clearPlaceholder();
        process.stdout.write(`\n[rcli-meet] answer failed: ${err.message}`);
      }
      process.stdout.write('\n');
      await speakAnswer(answer);
    } finally {
      answering = false;
    }
    while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
    rl.prompt(true);
  }

  const partials = { meeting: '', you: '' };
  const utteranceStart = { meeting: null, you: null };

  function wireStream(sttStream, source) {
    sttStream.on('partial', (text) => {
      if (speaking) return; // avoid showing/hearing our own TTS output
      if (utteranceStart[source] == null) utteranceStart[source] = transcript ? transcript.elapsedNow() : Date.now();
      partials[source] = text;
      if (answering) return;
      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      process.stdout.write(fitOneRow(`… [${source}] ${text}`));
    });

    sttStream.on('final', (text) => {
      const startedAt = utteranceStart[source] ?? (transcript ? transcript.elapsedNow() : Date.now());
      utteranceStart[source] = null;
      partials[source] = '';
      if (speaking) return; // our own voice, playing through the loopback -- ignore entirely

      const spokenCmd = parseCommand(text, { allowStop: false });
      if (spokenCmd) {
        void handleCommand(spokenCmd);
        return;
      }

      if (recording) {
        ensureSessionState();
        const seg = transcript.add(text, source, startedAt);
        retrieval.add(seg);
        summarizer.addSegment(seg);
        summarizer.maybeUpdate();
        if (answering) deferredLines.push(seg.line);
        else printLine(seg.line);
      } else if (source === 'you') {
        // Not recording: an utterance from your mic that isn't a command is a spoken question.
        printLine(`[you] ${text}`);
        void handleQuestion(text);
      }
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
        if (speaking) return; // don't feed our own TTS output back into recognition
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

  rl.on('line', (line) => void handleLine(line));

  async function handleLine(rawLine) {
    const line = rawLine.trim();
    if (!line) return rl.prompt();
    if (line === '/quit' || line === '/exit') return void shutdown();

    if (awaitingLoadChoice) {
      const names = awaitingLoadChoice;
      awaitingLoadChoice = null;
      return resolveLoadChoice(line, names);
    }

    const addMatch = /^add\s+(.+)$/i.exec(line);
    if (addMatch) return handleAddFile(addMatch[1].trim());

    const cmd = parseCommand(line, { allowStop: true });
    if (cmd) return void handleCommand(cmd);

    void handleQuestion(line);
  }

  let shuttingDown = false;
  async function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;
    // A command typed while an answer was still speaking gets its
    // confirmation deferred until the answer finishes -- /quit must not
    // race past that and exit before it's shown (the underlying action,
    // e.g. a file write from "save", already happened either way; only the
    // confirmation message was at risk of being lost).
    while (deferredLines.length) process.stdout.write(deferredLines.shift() + '\n');
    process.stdout.write('\n[rcli-meet] shutting down...\n');
    for (const source of Object.keys(captures)) captures[source].stop();
    for (const source of Object.keys(sttStreams)) sttStreams[source].close();
    stt.close();
    if (tts) tts.close();
    if (transcript) await transcript.close();
    engine.shutdown();
    rl.close();
    process.exit(0);
  }

  rl.on('SIGINT', () => void shutdown());
  process.on('SIGINT', () => void shutdown());

  console.log(`[rcli-meet] ready (window: ${opts.minutes} min, context budget: ${CONTEXT_TOKEN_BUDGET} tokens).`);
  console.log('[rcli-meet] Say or type "start" to begin recording. Type /quit to exit.\n');
  rl.prompt();
}

main().catch((err) => {
  console.log(`[rcli-meet] fatal error: ${err && err.message ? err.message : err}`);
  process.exit(1);
});
