// Rolling summary of the whole session, so "summarize the meeting" and other
// synthesis-style questions work past the point where the raw transcript no
// longer fits in the context window. Every SEGMENTS_PER_UPDATE finalized
// segments, folds the new lines into the existing summary via one more LLM
// call -- bounded output, so it stays cheap regardless of session length.
const { serialize, visibleOutside, NO_THINK_DIRECTIVE, TRANSCRIPTION_CAVEAT } = require('./llm');

const SEGMENTS_PER_UPDATE = Number(process.env.RCLI_MEET_SUMMARY_EVERY) || 8;
const SUMMARY_MAX_TOKENS = Number(process.env.RCLI_MEET_SUMMARY_TOKENS) || 220;
const SUMMARY_TEMPERATURE = 0.2;

function buildSummaryPrompt(existingSummary, newLines, disableThinking) {
  return `You maintain a running summary of a live meeting/call, for later Q&A. Update the summary to fold in the new transcript lines below. Keep it factual and concise -- a few sentences to a short paragraph, not a list of everything said. "[meeting]" lines are other people; "[you]" lines are the person this summary is for -- keep that distinction in the summary too (e.g. "you asked about X" vs "the team said Y").

${TRANSCRIPTION_CAVEAT}

Existing summary:
${existingSummary || '(none yet -- this is the first update)'}

New transcript lines:
${newLines.join('\n')}

Updated summary:${disableThinking ? `\n${NO_THINK_DIRECTIVE}` : ''}`;
}

/**
 * @param llm RunAnywhere LLMModel (already loaded)
 * @param disableThinking {boolean} from engine.disableThinking
 * @param onError {(message: string) => void}
 */
function createSummarizer({ llm, disableThinking = false, onError = () => {} }) {
  let summary = '';
  let pending = [];
  let running = false;

  return {
    get summary() {
      return summary;
    },
    get pendingCount() {
      return pending.length;
    },

    addSegment(segment) {
      pending.push(segment.line);
    },

    /**
     * Opportunistic, fire-and-forget: does nothing unless enough new lines
     * have accumulated and no update is already in flight. Safe to call
     * after every finalized segment -- actually running through `serialize`
     * means it naturally queues behind (or ahead of) any in-flight question
     * rather than racing the same LLM context.
     */
    maybeUpdate() {
      if (running || pending.length < SEGMENTS_PER_UPDATE) return;
      running = true;
      const batch = pending;
      pending = [];

      const prompt = buildSummaryPrompt(summary, batch, disableThinking);
      serialize(async () => {
        let raw = '';
        for await (const token of llm.generate(prompt, {
          maxTokens: SUMMARY_MAX_TOKENS,
          temperature: SUMMARY_TEMPERATURE,
        })) {
          raw += token;
        }
        const { text } = visibleOutside(raw);
        if (text.trim()) summary = text.trim();
      })
        .catch((err) => {
          onError(`meeting summary update failed: ${err.message}`);
          // Put the batch back so this content isn't silently lost from the
          // summary forever if it was a transient failure.
          pending = batch.concat(pending);
        })
        .finally(() => {
          running = false;
        });
    },
  };
}

module.exports = { createSummarizer, buildSummaryPrompt, SEGMENTS_PER_UPDATE, SUMMARY_MAX_TOKENS };
