"""Citation-grounded answer generation. Falls back to extractive mode without an API key."""
import os

MODEL = "claude-haiku-4-5-20251001"
SYSTEM = ("Answer ONLY from the numbered passages. Cite every claim like [1] or [2]. "
          "If the passages do not contain the answer, say you cannot find it.")


def format_passages(passages):
    return "\n\n".join(f"[{i}] ({p['doc']}, p.{p['page']}) {p['text']}"
                       for i, p in enumerate(passages, 1))


def ask_llm(prompt, system=SYSTEM, max_tokens=600):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": prompt}])
    return r.content[0].text


def answer(question, passages):
    if not os.getenv("ANTHROPIC_API_KEY"):
        top = passages[0]
        return f"(extractive mode, no API key) {top['text'][:400]}... [1]"
    return ask_llm(f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}")
