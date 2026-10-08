"""Hybrid retriever: BM25 + dense (FAISS) fused with RRF, optional cross-encoder rerank."""
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tok(s):
    return re.findall(r"\w+", s.lower())


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            scores[i] = scores.get(i, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, chunks, emb=None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
        self.emb = emb or SentenceTransformer(EMB_MODEL)
        vecs = self.emb.encode([c["text"] for c in chunks], normalize_embeddings=True,
                               show_progress_bar=False)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.asarray(vecs, dtype="float32"))
        self._reranker = None

    def _bm25(self, q, n):
        return [int(i) for i in np.argsort(self.bm25.get_scores(tok(q)))[::-1][:n]]

    def _dense(self, q, n):
        v = self.emb.encode([q], normalize_embeddings=True)
        _, idx = self.index.search(np.asarray(v, dtype="float32"), n)
        return [int(i) for i in idx[0] if i >= 0]

    def search(self, q, k=5, mode="hybrid", rerank=False, pool=30):
        if mode == "bm25":
            ids = self._bm25(q, pool)
        elif mode == "dense":
            ids = self._dense(q, pool)
        else:
            ids = rrf([self._bm25(q, pool), self._dense(q, pool)])[:pool]
        if rerank:
            if self._reranker is None:
                self._reranker = CrossEncoder(RERANK_MODEL)
            scores = self._reranker.predict([(q, self.chunks[i]["text"]) for i in ids])
            ids = [i for _, i in sorted(zip(scores, ids), reverse=True)]
        return [self.chunks[i] for i in ids[:k]]
