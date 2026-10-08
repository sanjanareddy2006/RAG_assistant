import json
import re


def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()


chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
texts = [norm(c["text"]) for c in chunks]

for i, line in enumerate(open("eval/qa.jsonl"), 1):
    if not line.strip():
        continue
    q = json.loads(line)
    n = sum(norm(q["gold_snippet"]) in t for t in texts)
    words = len(q["gold_snippet"].split())
    print(f"Q{i}: {'OK' if n else 'NOT FOUND'} ({n} chunk(s), {words} words)")


def avg_len(c):
    w = c["text"].split()
    return sum(map(len, w)) / max(len(w), 1)


bad = [c for c in chunks if avg_len(c) < 2.5]
print(f"\nPossibly garbled chunks: {len(bad)}")
for d in sorted({c["doc"] for c in bad}):
    print("  ", d)
