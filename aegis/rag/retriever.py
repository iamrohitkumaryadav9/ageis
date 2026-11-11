"""A small, auditable runbook retriever.

Corpus: versioned Markdown runbooks, split into passages at ``##`` headings.
Retrieval: TF-IDF cosine similarity (always available), optionally fused with
local embedding similarity (``nomic-embed-text`` through Ollama) by reciprocal
rank fusion. At this corpus size (tens of passages) a matrix in memory is the
whole index; a vector database would add a service and no recall.

Security: runbook text is untrusted data. Passages containing instruction-like
phrasing aimed at an assistant are flagged ``suspicious`` and are withheld from
LLM prompts. Nothing retrieved can call a tool or grant an approval; those
paths do not read runbook text at all.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |all )?(system|previous|prior)",
    r"you (are|must|should) (now )?(an?|the)? ?(assistant|ai|agent|model)",
    r"(as|dear) (an? )?(ai|assistant|agent|language model)",
    r"system prompt",
    r"\b(auto[- ]?approve|approve (this|the|all|every)|mark (it |this )?as approved|without (human )?approval)\b",
    r"\b(call|invoke|execute|run) the (tool|function)\b",
    r"\bskip (the )?(validation|approval|policy)\b",
    r"do not (tell|inform|alert) (the )?(user|operator|human)",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


class Passage(BaseModel):
    passage_id: str  # e.g. RB-004#remediation
    doc_id: str
    doc_title: str
    doc_version: str
    section: str
    text: str
    applies_to: list[str]
    sha256: str
    suspicious: bool = False


class RetrievedPassage(BaseModel):
    passage_id: str
    doc_id: str
    doc_title: str
    doc_version: str
    section: str
    text: str
    score: float
    method: str
    suspicious: bool = False


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


def _front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta: dict = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        meta[key.strip()] = (
            [v.strip() for v in value.strip("[]").split(",") if v.strip()]
            if value.startswith("[")
            else value
        )
    return meta, body


def load_passages(directory: Path) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        doc_id = str(meta.get("id") or path.stem)
        sections = re.split(r"^##\s+", body, flags=re.MULTILINE)
        for block in sections[1:]:
            title, _, text = block.partition("\n")
            text = " ".join(text.split())
            if not text:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            passages.append(
                Passage(
                    passage_id=f"{doc_id}#{slug}",
                    doc_id=doc_id,
                    doc_title=str(meta.get("title", path.stem)),
                    doc_version=str(meta.get("version", "0")),
                    section=title.strip(),
                    text=text,
                    applies_to=list(meta.get("applies_to", []))
                    if isinstance(meta.get("applies_to"), list)
                    else [],
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    suspicious=bool(_INJECTION.search(text) or _INJECTION.search(title)),
                )
            )
    return passages


@dataclass
class _EmbeddingIndex:
    matrix: np.ndarray  # row-normalised


class RunbookRetriever:
    def __init__(
        self,
        directory: Path,
        embedder: Embedder | None = None,
        cache_dir: Path | None = None,
        default_method: str = "lexical",
    ):
        self.passages = load_passages(directory)
        if not self.passages:
            raise ValueError(f"no runbook passages found in {directory}")
        self._texts = [f"{p.doc_title}. {p.section}. {p.text}" for p in self.passages]
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._tfidf.fit_transform(self._texts)
        self._embedder = embedder
        self.default_method = default_method
        self._cache_dir = cache_dir
        self._embeddings: _EmbeddingIndex | None = None
        self._embedding_tried = False

    @property
    def corpus_version(self) -> str:
        return hashlib.sha256("".join(p.sha256 for p in self.passages).encode()).hexdigest()[:12]

    # ------------------------------------------------------------ embeddings
    def _ensure_embeddings(self) -> _EmbeddingIndex | None:
        if self._embedding_tried or self._embedder is None:
            return self._embeddings
        self._embedding_tried = True
        cache = (
            self._cache_dir / f"runbook-embeddings-{self.corpus_version}.json"
            if self._cache_dir
            else None
        )
        vectors = None
        if cache and cache.exists():
            vectors = json.loads(cache.read_text())
        else:
            vectors = self._embedder.embed([f"search_document: {t}" for t in self._texts])
            if vectors and cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(vectors))
        if vectors:
            matrix = np.asarray(vectors, dtype=float)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
            self._embeddings = _EmbeddingIndex(matrix)
        return self._embeddings

    # ---------------------------------------------------------------- search
    def _lexical(self, query: str) -> np.ndarray:
        return (self._matrix @ self._tfidf.transform([query]).T).toarray().ravel()

    def _dense(self, query: str) -> np.ndarray | None:
        index = self._ensure_embeddings()
        if index is None or self._embedder is None:
            return None
        vector = self._embedder.embed([f"search_query: {query}"])
        if not vector:
            return None
        q = np.asarray(vector[0], dtype=float)
        return index.matrix @ (q / (np.linalg.norm(q) + 1e-12))

    def search(
        self, query: str, top_k: int = 3, method: str | None = None
    ) -> list[RetrievedPassage]:
        """``method``: lexical | dense | hybrid | auto (hybrid when embeddings exist, else lexical)."""
        method = method or self.default_method
        query = " ".join(str(query).split())[:500]
        lexical = self._lexical(query)
        dense = self._dense(query) if method in ("dense", "hybrid", "auto") else None
        if method == "dense" and dense is None:
            return []
        if dense is None or method == "lexical":
            scores, used = lexical, "lexical"
        elif method == "dense":
            scores, used = dense, "dense"
        else:  # reciprocal rank fusion
            scores = np.zeros(len(self.passages))
            for ranking in (np.argsort(-lexical), np.argsort(-dense)):
                scores[ranking] += 1.0 / (60 + np.arange(1, len(ranking) + 1))
            used = "hybrid"
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedPassage(
                **self.passages[i].model_dump(
                    include={
                        "passage_id",
                        "doc_id",
                        "doc_title",
                        "doc_version",
                        "section",
                        "text",
                        "suspicious",
                    }
                ),
                score=round(float(scores[i]), 5),
                method=used,
            )
            for i in order
            if scores[i] > 0
        ]


def evaluate_retrieval(
    retriever: RunbookRetriever, queries: list[dict], method: str, k: int = 3
) -> dict:
    """Document-level hit@1, hit@k and MRR for labelled queries."""
    hits1 = hitsk = 0
    reciprocal = 0.0
    answered = 0
    for item in queries:
        results = retriever.search(item["query"], top_k=10, method=method)
        if not results:
            continue
        answered += 1
        docs = list(dict.fromkeys(r.doc_id for r in results))
        relevant = set(item["relevant"])
        hits1 += docs[0] in relevant
        hitsk += bool(relevant & set(docs[:k]))
        rank = next((i + 1 for i, d in enumerate(docs) if d in relevant), None)
        reciprocal += 1.0 / rank if rank else 0.0
    n = len(queries)
    return {
        "method": method,
        "queries": n,
        "answered": answered,
        "hit@1": round(hits1 / n, 4),
        f"hit@{k}": round(hitsk / n, 4),
        "mrr": round(reciprocal / n, 4),
    }
