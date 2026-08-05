// Loads RunAnywhere's engine (Vulkan-accelerated LLM on the 6800XT, already
// proven working in this SDK) and answers questions grounded in the rolling
// transcript.
const ELECTRON_SDK_DIST =
  process.env.RCLI_MEET_SDK_DIST ||
  'D:/the_code/runanywhere/SDK/runanywhere-sdks-main/sdk/runanywhere-electron/dist';
const { RunAnywhere } = require(ELECTRON_SDK_DIST);

// Any RunAnywhere LLM catalog id (e.g. "qwen2.5-3b", auto-downloaded) or a
// local GGUF path works here; override with RCLI_MEET_LLM_PATH.
const DEFAULT_LLM_PATH = process.env.RCLI_MEET_LLM_PATH || 'qwen2.5-3b';
const DEFAULT_EMBEDDER_ID = process.env.RCLI_MEET_EMBEDDER_ID || 'minilm';

async function loadEngine({ llmPath = DEFAULT_LLM_PATH, embedderId = DEFAULT_EMBEDDER_ID } = {}) {
  RunAnywhere.initialize({ environment: 'development' });
  const llm = await RunAnywhere.loadLLM(llmPath);
  const embedder = await RunAnywhere.loadEmbedder(embedderId);

  return {
    llm,
    embedder,
    shutdown() {
      llm.unload();
      embedder.unload();
      RunAnywhere.shutdown();
    },
  };
}

function buildPrompt({ windowText, retrieved, question }) {
  const retrievedText = retrieved.map((r) => r.line).join('\n');
  return `You are answering questions about a live meeting/talk transcript. Use ONLY the transcript excerpts below. If the answer isn't in them, say so briefly. Be concise.

Relevant earlier moments:
${retrievedText || '(none)'}

Recent transcript:
${windowText || '(none yet)'}

Question: ${question}
Answer:`;
}

async function askQuestion(llm, promptCtx, onToken) {
  const prompt = buildPrompt(promptCtx);
  let out = '';
  for await (const token of llm.generate(prompt, { maxTokens: 200, temperature: 0.3 })) {
    out += token;
    onToken(token);
  }
  return out;
}

module.exports = { loadEngine, buildPrompt, askQuestion, DEFAULT_LLM_PATH, DEFAULT_EMBEDDER_ID };
