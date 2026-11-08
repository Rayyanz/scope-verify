"""Dense retrieval over a per-question paragraph pool (E5 via sentence-transformers)."""
import numpy as np


class PoolRetriever:
    def __init__(self, model_name="intfloat/e5-base-v2", device=None, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._cache = {}

    def _embed(self, texts, prefix):
        missing = [t for t in texts if (prefix, t) not in self._cache]
        if missing:
            vecs = self.model.encode([prefix + t for t in missing], batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[(prefix, t)] = v
        return np.stack([self._cache[(prefix, t)] for t in texts])

    def search(self, query, pool, k=3):
        """Return the top-k paragraphs of `pool` (list of dicts) for `query`."""
        if not pool:
            return []
        P = self._embed([p["title"] + ". " + p["text"] for p in pool], "passage: ")
        q = self._embed([query], "query: ")[0]
        order = np.argsort(-(P @ q))[:k]
        return [pool[i] for i in order]
