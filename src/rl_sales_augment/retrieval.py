"""Retrieval hook: give the agent a company knowledge base bigger than one prompt.

The agent accepts any `retrieve_fn(query, move) -> list[str]`. Retrieval is MOVE-CONDITIONED:
the RL-chosen move shapes what gets fetched (an OBJECTION turn wants rebuttals and proof, a
PITCH turn wants product specs, a CLOSE turn wants terms and next steps). Retrieved chunks are
injected into the system prompt AND join the grounding sources, so a price fetched from the
knowledge base can be quoted, while a price from nowhere still gets caught.

`SimpleRetriever` is the zero-dependency reference: keyword-overlap scoring over text chunks.
For real deployments plug your own vector store; anything with the same callable shape works:

    kb = rsa.SimpleRetriever.from_texts([open("catalog.txt").read(), open("faq.txt").read()])
    bot = rsa.load_agent(gen, company_ctx=CORE_FACTS, retrieve_fn=kb)
"""
from __future__ import annotations
import re

# what each strategic move wants from the knowledge base
MOVE_QUERY_HINTS = {
    "RESEARCH": "customer fit use case requirements",
    "RAPPORT": "company story customers social proof",
    "PITCH": "product features benefits specifications",
    "OBJECTION": "objection concern rebuttal proof comparison guarantee",
    "DISCOUNT": "pricing plans price cost discount tiers",
    "FOLLOWUP": "next step demo trial timeline",
    "CLOSE": "terms onboarding contract getting started payment",
    "DROP": "alternatives referral",
}

_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "you", "your", "with", "that", "this", "are", "have", "can",
         "our", "will", "what", "how", "about", "from", "just", "like", "get", "not"}


def _tokens(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def chunk_text(text, size=380, overlap=60):
    """Split a document into overlapping chunks on sentence-ish boundaries."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            dot = text.rfind(". ", start + size // 2, end)
            if dot > 0:
                end = dot + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 40]


class SimpleRetriever:
    """Zero-dependency keyword retriever over chunks. Callable as retrieve_fn(query, move)."""

    def __init__(self, chunks, k=3):
        self.chunks = list(chunks)
        self.k = k
        self._toks = [set(_tokens(c)) for c in self.chunks]

    @classmethod
    def from_texts(cls, texts, k=3, chunk_size=380):
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t, size=chunk_size))
        return cls(chunks, k=k)

    def __call__(self, query, move=None):
        q = set(_tokens(query))
        if move:
            q |= set(_tokens(MOVE_QUERY_HINTS.get(move, "")))
        if not q:
            return []
        scored = sorted(((len(q & toks), i) for i, toks in enumerate(self._toks)), reverse=True)
        return [self.chunks[i] for score, i in scored[: self.k] if score > 0]
