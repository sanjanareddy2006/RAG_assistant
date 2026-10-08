import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from rag.retriever import Retriever
from rag.generate import answer

state = {}


@asynccontextmanager
async def lifespan(app):
    chunks = [json.loads(l) for l in open("data/chunks.jsonl")]
    state["r"] = Retriever(chunks)
    yield


app = FastAPI(title="LexRAG", lifespan=lifespan)


class Query(BaseModel):
    question: str
    k: int = 5
    rerank: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["r"].chunks)}


@app.post("/ask")
def ask(q: Query):
    passages = state["r"].search(q.question, k=q.k, mode="hybrid", rerank=q.rerank)
    return {"answer": answer(q.question, passages),
            "sources": [{"doc": p["doc"], "page": p["page"], "text": p["text"]} for p in passages]}
