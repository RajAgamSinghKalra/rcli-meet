// Cosine-similarity search over every transcript segment for the whole
// session (not just the recency window) -- lets a question pull in a moment
// from long before the last-N-minutes window, e.g. "combine what she said at
// the start with what he said just now".
function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function createRetrieval(embedder) {
  const items = [];

  return {
    /** @param segment {{line: string, text: string, elapsedMs: number}} */
    add(segment) {
      const vec = embedder.embed(segment.text);
      items.push({ ...segment, vec });
    },

    /** Top-k segments by cosine similarity to `query` (embeddings are already L2-normalized). */
    topK(query, k = 5) {
      if (items.length === 0) return [];
      const qvec = embedder.embed(query);
      return items
        .map((item) => ({ item, score: dot(qvec, item.vec) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k)
        .map((s) => s.item);
    },
  };
}

module.exports = { createRetrieval };
