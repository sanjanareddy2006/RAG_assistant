# """Load PDFs/text from data/raw, chunk them, save to data/chunks.jsonl."""
# import json, re, sys
# from pathlib import Path
# from pypdf import PdfReader


# def load_docs(raw_dir="data/raw"):
#     docs = []
#     for p in sorted(Path(raw_dir).glob("**/*")):
#         suf = p.suffix.lower()
#         if suf == ".pdf":
#             for i, page in enumerate(PdfReader(str(p)).pages, 1):
#                 docs.append({"doc": p.name, "page": i, "text": page.extract_text() or ""})
#         elif suf in (".txt", ".md"):
#             docs.append({"doc": p.name, "page": 1, "text": p.read_text(errors="ignore")})
#     return docs


# def chunk_docs(docs, size=300, overlap=50):
#     """Sliding word window. size/overlap are the knobs you ablate in eval."""
#     chunks, step = [], max(size - overlap, 1)
#     for d in docs:
#         words = re.sub(r"\s+", " ", d["text"]).split()
#         for start in range(0, len(words), step):
#             piece = words[start:start + size]
#             if len(piece) < 20:
#                 continue
#             chunks.append({"id": len(chunks), "doc": d["doc"], "page": d["page"],
#                            "text": " ".join(piece)})
#     return chunks


# if __name__ == "__main__":
#     size = int(sys.argv[1]) if len(sys.argv) > 1 else 300
#     chunks = chunk_docs(load_docs(), size=size)
#     with open("data/chunks.jsonl", "w") as f:
#         for c in chunks:
#             f.write(json.dumps(c) + "\n")
#     print(f"Wrote {len(chunks)} chunks (size={size})")

"""Load PDFs/text from data/raw, clean headers/footers, chunk, save to data/chunks.jsonl."""
import collections
import json
import re
import sys
from pathlib import Path
from pypdf import PdfReader


def _norm(line):
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages):
    """Remove site watermark and short lines repeated across many pages (running headers/footers)."""
    split = [p.splitlines() for p in pages]
    counts = collections.Counter()
    for lines in split:
        for n in {_norm(l) for l in lines if l.strip()}:
            counts[n] += 1
    thresh = max(3, int(0.3 * len(pages)))
    cleaned = []
    for lines in split:
        keep = [l for l in lines
                if l.strip()
                and "indiankanoon" not in l.lower()
                and not (counts[_norm(l)] >= thresh and len(l) < 150)]
        cleaned.append("\n".join(keep))
    return cleaned


def load_docs(raw_dir="data/raw"):
    docs = []
    for p in sorted(Path(raw_dir).glob("**/*")):
        suf = p.suffix.lower()
        if suf == ".pdf":
            pages = [pg.extract_text() or "" for pg in PdfReader(str(p)).pages]
            pages = clean_pages(pages)
            if sum(len(t) for t in pages) < 200:
                print(f"WARNING: no extractable text (scanned?): {p.name}")
                continue
            for i, text in enumerate(pages, 1):
                docs.append({"doc": p.name, "page": i, "text": text})
        elif suf in (".txt", ".md"):
            docs.append({"doc": p.name, "page": 1,
                        "text": p.read_text(errors="ignore")})
    return docs


# def chunk_docs(docs, size=300, overlap=50):
#     """Sliding word window. size/overlap are the knobs you ablate in eval."""
#     chunks, step = [], max(size - overlap, 1)
#     for d in docs:
#         words = re.sub(r"\s+", " ", d["text"]).split()
#         for start in range(0, len(words), step):
#             piece = words[start:start + size]
#             if len(piece) < 20:
#                 continue
#             chunks.append({"id": len(chunks), "doc": d["doc"], "page": d["page"],
#                            "text": " ".join(piece)})
#     return chunks

def chunk_docs(docs, size=300, overlap=50):
    """Sliding word window. Skips garbled (letter-spaced) and exact-duplicate chunks."""
    chunks, step, seen = [], max(size - overlap, 1), set()
    for d in docs:
        words = re.sub(r"\s+", " ", d["text"]).split()
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            if len(piece) < 20:
                continue
            if sum(map(len, piece)) / len(piece) < 2.5:   # garbled extraction
                continue
            text = " ".join(piece)
            if text in seen:                               # exact duplicate
                continue
            seen.add(text)
            chunks.append(
                {"id": len(chunks), "doc": d["doc"], "page": d["page"], "text": text})
    return chunks


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    chunks = chunk_docs(load_docs(), size=size)
    with open("data/chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(chunks)} chunks (size={size})")
