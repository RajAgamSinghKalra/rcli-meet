#!/usr/bin/env node
// rcli-meet: offline live-meeting captions + local-LLM Q&A over the rolling
// transcript, with voice commands, TTS replies, and save/load. CLI only, no
// UI. Everything runs locally. Nothing leaves the machine.
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const { startCapture } = require('./capture');
// Default STT is Vulkan whisper.cpp (GPU on the 6800XT) -- RunAnywhere's own
// STT path is CPU-only sherpa, and the old npm `sherpa-onnx` dependency was
// WebAssembly, which made captions both slow and inaccurate. CPU fallbacks:
//   sensevoice -- FunASR SenseVoice (strong on accented English, native CPU)
//   whisper    -- sherpa Whisper-small (native CPU, not WASM)
//   zipformer  -- streaming partials (weak on Indian English)
const STT_ENGINE = (process.env.RCLI_MEET_STT_ENGINE || 'vulkan').toLowerCase();
const STT_MODULES = {
  vulkan: './sttVulkan',
  sensevoice: './sttSenseVoice',
  whisper: './sttWhisper',
  zipformer: './stt',
};
if (!STT_MODULES[STT_ENGINE]) {
  console.log(
    `[rcli-meet] unknown RCLI_MEET_STT_ENGINE="${STT_ENGINE}" (want: vulkan|sensevoice|whisper|zipformer)`
  );
  process.exit(2);
}
const { createSTTEngine, assertModelPresent } = require(STT_MODULES[STT_ENGINE]);
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

const DEFAULT_MODEL_DIRS = {
  vulkan: path.join(__dirname, '..', 'models', 'ggml-large-v3-turbo.bin'),
  sensevoice: path.join(
    __dirname,
    '..',
    'models',
    'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17'
  ),
  whisper: path.join(__dirname, '..', 'models', 'sherpa-onnx-whisper-small.en'),
  zipformer: path.join(__dirname, '..', 'models', 'sherpa-onnx-streaming-zipformer-en-2023-06-26'),
};
const MODEL_DIR = process.env.RCLI_MEET_STT_MODEL_DIR || DEFAULT_MODEL_DIRS[STT_ENGINE];
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
  --mic <name>         Substring to pick a mic (e.g. "Razer" or "Kraken")
  --mic-gain <n>       Software mic gain (default 1; try 4–8 if the mic is quiet)
  --no-tts             Don't speak answers out loud
  -h, --help           Show this help

In-session (typed OR spoken, except stop which is typed-only):
  start / record       Begin recording — transcribes meeting + mic into a session folder
  stop                 (typed only) Stop recording → voice chat mode (mic only, spoken replies)
  save                 Save the current session to sessions/<date>_<app>/
  load                 List saved sessions and pick one to load into context
  add <path>           Ingest a .txt file into the current session (typed only)
  anything else        Ask a question — answered from the transcript, spoken back
  /quit                Exit

Environment: see README.md (RCLI_MEET_SDK_DIST, RCLI_MEET_LLM_PATH, RCLI_MEET_STT_ENGINE=vulkan|sensevoice|whisper|zipformer, RCLI_MEET_PYTHON, etc.)

Note: without headphones, your microphone may also pick up the meeting audio
itself (speaker bleed), which would mislabel meeting speech as "you" said it.`;

class UsageError extends Error {}

function parseArgs(argv) {
  const opts = {
    minutes: 20,
    llmPath: DEFAULT_LLM_PATH,
    embedderId: DEFAULT_EMBEDDER_ID,
    mic: true,
    tts: true,
    micName: process.env.RCLI_MEET_MIC || '',
    micGain: Number(process.env.RCLI_MEET_MIC_GAIN || '1') || 1,
  };
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
    else if (arg === '--mic') opts.micName = takeValue(arg, i++);
    else if (arg === '--mic-gain') {
      const raw = takeValue(arg, i++);
      const gain = Number(raw);
      if (!Number.isFinite(gain) || gain <= 0) throw new UsageError(`--mic-gain must be a positive number (got "${raw}")`);
      opts.micGain = gain;
    } else if (arg === '--no-tts') opts.tts = false;
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

  console.log(
    STT_ENGINE === 'vulkan'
      ? '[rcli-meet] loading Vulkan Whisper STT (GPU large-v3-turbo on RX 6800XT)...'
      : `[rcli-meet] loading STT engine (${STT_ENGINE})...`
  );
  const stt = createSTTEngine(MODEL_DIR);
  if (typeof stt.ready === 'function') {
    console.log('[rcli-meet] waiting for GPU whisper worker (model load)...');
    await stt.ready();
  }
  console.log(`[rcli-meet] STT ready (engine: ${STT_ENGINE}).`);

  const tts = opts.tts ? createTTS() : null;
  console.log(opts.tts ? '[rcli-meet] TTS ready -- answers will be spoken.' : '[rcli-meet] TTS disabled (--no-tts).');
  if (opts.mic) {
    console.log(
      '[rcli-meet] mic capture enabled' +
        (opts.micName ? ` (device filter: "${opts.micName}")` : ' (Windows default input)') +
        (opts.micGain !== 1 ? `, gain×${opts.micGain}` : '') +
        '.'
    );
    console.log(
      '[rcli-meet] If mic stays silent: unmute the Kraken dial/boom, check Razer Synapse + Windows input volume.'
    );
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
    terminal: !!process.stdin.isTTY,
  });

  let answering = false;
  let speaking = false; // true while TTS plays
  let muteUntil = 0; // post-TTS cooldown (loopback + mic echo)
  let lastSpokenText = ''; // echo-cancel meeting captions of our own TTS
  let recording = false;
  let sessionDir = null; // survives stop(); replaced on the next start()
  let transcript = null;
  let retrieval = null;
  let summarizer = null;
  let awaitingLoadChoice = null; // string[] of session names, set right after "load" lists them
  const deferredLines = [];
  const partials = { meeting: '', you: '' };
  const utteranceStart = { meeting: null, you: null };
  const sttStreams = {};
  const captures = {};

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
      // Drop any stale meeting audio that was buffered while we were in chat mode.
      if (sttStreams.meeting && typeof sttStreams.meeting.reset === 'function') {
        sttStreams.meeting.reset();
      }
      recording = true;
      notify(
        `recording started (window: "${appName}"). Capturing meeting + mic into ${sessionDir}`
      );
    } else if (cmd === 'stop') {
      if (!recording) return notify('not currently recording.');
      recording = false;
      if (sttStreams.meeting && typeof sttStreams.meeting.reset === 'function') {
        sttStreams.meeting.reset();
      }
      partials.meeting = '';
      utteranceStart.meeting = null;
      if (opts.mic) {
        notify(
          'recording stopped — voice chat mode. Meeting audio ignored; speak into your mic and I\'ll answer out loud. Type "start" to record again, or "save".'
        );
      } else {
        notify(
          'recording stopped — meeting audio ignored. Mic is disabled (--no-mic); type questions, or "start" / "save".'
        );
      }
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

  function resetSttBuffers() {
    for (const source of Object.keys(sttStreams)) {
      if (typeof sttStreams[source].reset === 'function') sttStreams[source].reset();
      partials[source] = '';
      utteranceStart[source] = null;
    }
  }

  /** Mute capture while we (or our echo) could be on the speakers. */
  function isAudioMuted(source) {
    if (speaking || Date.now() < muteUntil) return true;
    // While answering with TTS enabled, keep meeting loopback deaf — otherwise
    // the assistant's own voice is captioned as [meeting] once playback starts.
    if (source === 'meeting' && answering && opts.tts) return true;
    return false;
  }

  function normalizeEcho(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /** True if meeting text is basically our own spoken answer leaking via loopback. */
  function isEchoOfLastSpoken(text) {
    const a = normalizeEcho(lastSpokenText);
    const b = normalizeEcho(text);
    if (!a || !b || a.length < 8) return false;
    if (a === b) return true;
    if (a.includes(b) || b.includes(a)) return true;
    // Share enough words with the last answer
    const aw = new Set(a.split(' ').filter((w) => w.length > 2));
    const bw = b.split(' ').filter((w) => w.length > 2);
    if (!bw.length) return false;
    let hit = 0;
    for (const w of bw) if (aw.has(w)) hit++;
    return hit / bw.length >= 0.6;
  }

  async function speakAnswer(text) {
    if (!tts || !text) return;
    lastSpokenText = text;
    speaking = true;
    // Drop anything already buffered (loopback of prior audio / mic bleed).
    resetSttBuffers();
    try {
      await tts.speak(text);
    } catch (err) {
      notify(`TTS playback failed: ${err.message}`);
    } finally {
      speaking = false;
      resetSttBuffers();
      // Razer loopback keeps ringing for a bit after playback ends.
      muteUntil = Date.now() + 2200;
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

  function wireStream(sttStream, source) {
    sttStream.on('error', (err) => {
      notify(`${source} transcription error: ${err && err.message ? err.message : err}`);
    });

    sttStream.on('partial', (text) => {
      if (isAudioMuted(source)) return;
      // Chat mode: only the mic is active — ignore meeting captions entirely.
      if (!recording && source === 'meeting') return;
      if (source === 'meeting' && isEchoOfLastSpoken(text)) return;
      if (utteranceStart[source] == null) {
        utteranceStart[source] = transcript ? transcript.elapsedNow() : Date.now();
      }
      partials[source] = text;
      if (answering) return;
      readline.clearLine(process.stdout, 0);
      readline.cursorTo(process.stdout, 0);
      const tag = !recording && source === 'you' ? 'you → chat' : source;
      process.stdout.write(fitOneRow(`… [${tag}] ${text}`));
    });

    sttStream.on('final', (text) => {
      const startedAt = utteranceStart[source] ?? (transcript ? transcript.elapsedNow() : Date.now());
      utteranceStart[source] = null;
      partials[source] = '';
      if (isAudioMuted(source)) return;
      if (!recording && source === 'meeting') return;
      if (source === 'meeting' && isEchoOfLastSpoken(text)) {
        notify('(ignored meeting echo of my own voice)');
        return;
      }

      const spokenCmd = parseCommand(text, { allowStop: false });
      if (spokenCmd) {
        notify(`heard command "${spokenCmd}" (from: "${text}")`);
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
        // Voice chat mode (not recording): mic utterance → question → spoken answer.
        printLine(`[you] ${text}`);
        void handleQuestion(text);
      }
    });
  }

  for (const source of SOURCES) {
    if (source === 'you' && !opts.mic) continue;
    const sttStream = stt.createStream();
    wireStream(sttStream, source);
    sttStreams[source] = sttStream;
    captures[source] = startCapture(
      source === 'meeting' ? 'loopback' : 'mic',
      (samples) => {
        if (isAudioMuted(source)) return;
        // While not recording, don't waste GPU on meeting loopback — chat is mic-only.
        if (!recording && source === 'meeting') return;
        try {
          sttStream.feed(samples);
        } catch (err) {
          notify(`${source} transcription error: ${err.message}`);
        }
      },
      (message) => {
        notify(message);
        notify(`${source} captions have stopped; other sources keep working.`);
      },
      source === 'you' ? { device: opts.micName, gain: opts.micGain } : {}
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
  if (opts.mic) {
    console.log(
      '[rcli-meet] Voice chat is on (mic only). Speak a question and I\'ll answer out loud.'
    );
    console.log('[rcli-meet] Say or type "start" to record the meeting. Type "stop" to return to chat. /quit to exit.\n');
  } else {
    console.log('[rcli-meet] Mic disabled. Type questions, or "start" to record meeting audio only. /quit to exit.\n');
  }
  rl.prompt();
}

main().catch((err) => {
  console.log(`[rcli-meet] fatal error: ${err && err.message ? err.message : err}`);
  process.exit(1);
});
