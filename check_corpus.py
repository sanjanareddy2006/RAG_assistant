import json
import collections

chunks = [json.loads(line) for line in open("data/chunks.jsonl")]
per_doc = collections.Counter(c["doc"] for c in chunks)

print(f"{len(chunks)} chunks from {len(per_doc)} documents (expected 41)\n")

print("Smallest 5 documents (tiny = possible failed extraction):")
for doc, n in per_doc.most_common()[-5:]:
    print(f"  {n:4d} chunks  {doc}")

kk = sum("indiankanoon" in c["text"].lower() for c in chunks)
print(
    f"\nChunks mentioning 'indiankanoon': {kk} (high number = boilerplate to clean)")

print("\nSample chunk:\n", chunks[len(chunks) // 2]["text"][:400])
