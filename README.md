# LexRAG: citation-grounded Q&A with a measured evaluation harness

Hybrid retrieval (BM25 + dense + reranker) over a domain corpus, answers with `[n]` citations,
and an ablation harness so every design choice is backed by a number.

## Quickstart
```bash
pip install -r requirements.txt
# 1. put PDFs/txt in data/raw/   (e.g. Supreme Court judgments, policy docs, guidelines)
python -m rag.ingest 300            # -> data/chunks.jsonl
export ANTHROPIC_API_KEY=...        # optional; without it, extractive fallback is used
uvicorn app:app --reload            # POST /ask {"question": "..."}
# 2. write 50-100 items in eval/qa.jsonl, then:
python -m eval.run_eval --faithfulness
```

## Results (fill from eval/results.md)
| chunk size | retrieval | rerank | hit@5 | MRR | faithful |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

## Failure analysis
(3-5 concrete questions it still gets wrong, and why: tables, cross-references, long-range context.)

## Deploy
Hugging Face Spaces (Docker SDK) or Render: push this repo, set `ANTHROPIC_API_KEY` as a secret.
Commit a small `data/chunks.jsonl` so the Space starts without re-ingesting.

## 3-week plan
- Week 1: pick corpus, ingest, baseline BM25, write 30 eval questions.
- Week 2: dense + hybrid + rerank, run ablations, finish 50-100 questions, add faithfulness check.
- Week 3: Streamlit/HTML front end, Docker, deploy, README table, demo GIF, failure analysis.
