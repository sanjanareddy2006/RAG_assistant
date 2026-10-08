"""Ablation harness: chunk size x retrieval mode x rerank -> hit@k, MRR, (optional) faithfulness.

Usage: python -m eval.run_eval [--k 5] [--faithfulness]
gold_snippet is matched against retrieved chunk text (whitespace/case-normalised),
so keep it short enough to not straddle a chunk boundary.
"""
import argparse, json, re
from sentence_transformers import SentenceTransformer
from rag.ingest import load_docs, chunk_docs
from rag.retriever import Retriever, EMB_MODEL
from rag.generate import answer, ask_llm, format_passages

norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
CONFIGS = [("bm25", False), ("dense", False), ("hybrid", False), ("hybrid", True)]
SIZES = [150, 300, 500]


def judge(question, ans, passages):
    prompt = (f"Passages:\n{format_passages(passages)}\n\nAnswer: {ans}\n\n"
              "Is every factual claim in the answer supported by the passages? Reply YES or NO only.")
    return ask_llm(prompt, system="You are a strict grader.", max_tokens=5).strip().upper().startswith("YES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--faithfulness", action="store_true")
    a = ap.parse_args()

    qa = [json.loads(l) for l in open("eval/qa.jsonl")]
    docs, emb = load_docs(), SentenceTransformer(EMB_MODEL)
    rows = ["| chunk size | retrieval | rerank | hit@%d | MRR |%s" % (a.k, " faithful |" if a.faithfulness else ""),
            "|---|---|---|---|---|" + ("---|" if a.faithfulness else "")]
    for size in SIZES:
        r = Retriever(chunk_docs(docs, size=size), emb=emb)
        for mode, rerank in CONFIGS:
            hits, rr, faith = 0, 0.0, 0
            for item in qa:
                res = r.search(item["question"], k=a.k, mode=mode, rerank=rerank)
                pos = next((i for i, c in enumerate(res) if norm(item["gold_snippet"]) in norm(c["text"])), None)
                if pos is not None:
                    hits += 1
                    rr += 1 / (pos + 1)
                if a.faithfulness:
                    faith += judge(item["question"], answer(item["question"], res), res)
            n = len(qa)
            row = f"| {size} | {mode} | {'yes' if rerank else 'no'} | {hits/n:.2f} | {rr/n:.2f} |"
            if a.faithfulness:
                row += f" {faith/n:.2f} |"
            rows.append(row)
            print(row, flush=True)
    open("eval/results.md", "w").write("\n".join(rows) + "\n")
    print("\nSaved eval/results.md. Paste this table into your README.")


if __name__ == "__main__":
    main()
