import json
import random

chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
random.seed(7)  # change the seed for a different sample
for c in random.sample(chunks, 12):
    print(f"--- {c['doc']} (page {c['page']}) ---")
    print(c["text"][:900], "\n")
