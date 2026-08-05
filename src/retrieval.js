// Cosine-similarity search over every transcript segment for the whole
// session (not just the recency window) -- lets a question pull in a moment
// from long before the last-N-minutes window, e.g. "combine what she said at
// the start with what he said just now".
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * @param embedder RunAnywhere embedder (embed() returns an L2-normalized Float32Array)
 * @param onError {(message: string) => void} non-fatal embed failures
 */
function createRetrieval(embedder, { onError = () => {} } = {}) {
  const items = [];

  return {
    /**
     * @param segment {{line: string, text: string, elapsedMs: number}}
     * @returns {boolean} whether the segment was indexed
     */
    add(segment) {
      // embed() runs native code; a failure here used to propagate out of the
      // STT 'final' handler and kill the session. Losing one segment from the
      // similarity index is far better -- it's still in the transcript and
      // still reachable through the recency window.
      try {
        const vec = embedder.embed(segment.text);
        items.push({ ...segment, vec });
        return true;
      } catch (err) {
        onError(`could not index a transcript segment for search: ${err.message}`);
        return false;
      }
    },

    get size() {
      return items.length;
    },

    /**
     * Top-k segments by cosine similarity to `query` (embeddings are already
     * L2-normalized, so a dot product IS the cosine similarity).
     * @param exclude {Set<string>} segment lines to omit (e.g. already shown
     *   verbatim in the recency window) so the prompt doesn't repeat itself.
     */
    topK(query, k = 5, exclude = new Set()) {
      if (items.length === 0) return [];
      let qvec;
      try {
        qvec = embedder.embed(query);
      } catch (err) {
        onError(`could not embed the question for search: ${err.message}`);
        return [];
      }
      return items
        .filter((item) => !exclude.has(item.line))
        .map((item) => ({ item, score: dot(qvec, item.vec) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k)
        .map((s) => s.item);
    },
  };
}

module.exports = { createRetrieval, dot };
