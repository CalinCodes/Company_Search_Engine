"""
Stage 2: Candidate Retrieval

Takes the hard-filtered candidates from Stage 1 (processed1.json) and re-ranks
them using a hybrid score:

    stage2_score = α * bm25_norm  +  β * embed_sim

BM25 is always active. Embeddings are optional (set USE_EMBEDDINGS = False to
skip them — useful during development or on low-resource machines).

Output: processed2.json — same shape as processed1.json but companies are
sorted by `stage2_score` descending and a `_retrieval` annotation is attached
to each record.
"""

import json
import math
import re
from typing import Any

# ── Optional embedding support ────────────────────────────────────────────────
USE_EMBEDDINGS = True          # flip to False to use BM25-only
EMBED_MODEL    = "all-MiniLM-L6-v2"   # ~80 MB, fast, good quality
ALPHA          = 0.6           # weight for BM25 normalised score
BETA           = 0.4           # weight for embedding cosine similarity

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False
    USE_EMBEDDINGS   = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    raise ImportError("rank-bm25 is required: pip install rank-bm25")


# ── Text helpers ──────────────────────────────────────────────────────────────

def _to_list(val: Any) -> list:
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if isinstance(val, dict):
        return " ".join(str(v) for v in val.values())
    return str(val)


def build_company_text(co: dict) -> str:
    """Concatenate all semantic fields into a single searchable string."""
    parts = [
        _safe_str(co.get("operational_name")),
        _safe_str(co.get("description")),
        _safe_str(co.get("primary_naics_label")),
        # secondary_naics can be a dict or list of dicts
        " ".join(
            item.get("label", "") if isinstance(item, dict) else str(item)
            for item in _to_list(co.get("secondary_naics"))
        ),
        _safe_str(co.get("core_offerings")),
        _safe_str(co.get("target_markets")),
        _safe_str(co.get("business_model")),
    ]
    return " ".join(p for p in parts if p).lower()


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index(companies: list[dict]) -> tuple[BM25Okapi, list[str]]:
    corpus_texts = [build_company_text(co) for co in companies]
    tokenized    = [tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized), corpus_texts


def bm25_scores(index: BM25Okapi, query_tokens: list[str]) -> list[float]:
    raw = index.get_scores(query_tokens)
    # BM25 raw values can be negative depending on corpus/query statistics.
    # Min-max scaling keeps the score range stable in [0, 1].
    min_score = float(min(raw))
    max_score = float(max(raw))
    if max_score == min_score:
        return [1.0] * len(raw)
    scale = max_score - min_score
    return [(float(s) - min_score) / scale for s in raw]


# ── Embeddings ────────────────────────────────────────────────────────────────

_model_cache: SentenceTransformer | None = None


def _get_model() -> "SentenceTransformer":
    global _model_cache
    if _model_cache is None:
        print(f"Loading embedding model '{EMBED_MODEL}' …")
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache


def embed_texts(texts: list[str]) -> "np.ndarray":
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def cosine_sim_matrix(query_vec: "np.ndarray", corpus_vecs: "np.ndarray") -> list[float]:
    """query_vec: (dim,), corpus_vecs: (N, dim) — embeddings already L2-normalised."""
    sims = corpus_vecs @ query_vec          # dot product == cosine sim when normalised
    return sims.tolist()


# ── Build query text ─────────────────────────────────────────────────────────

def build_query_text(query_parsed: dict) -> str:
    """
    Combine the raw query, semantic_keywords, and role_label into a single
    retrieval query string for maximum recall.
    """
    raw      = query_parsed.get("original_query") or ""
    keywords = _safe_str(query_parsed.get("semantic_keywords"))
    role     = query_parsed.get("role_label") or ""
    return f"{raw} {keywords} {role}".strip()


# ── Core run function ─────────────────────────────────────────────────────────

def run(
    query_parsed: dict,
    input_path:   str = "processed1.json",
    output_path:  str = "processed2.json",
    top_k:        int | None = None,
) -> list[dict]:
    """
    Load Stage 1 output, score each company using BM25 (+ optionally
    embeddings), and write ranked results to processed2.json.

    Args:
        query_parsed:  The dict returned by stage1_parser.parse_query().
        input_path:    Path to Stage 1 output JSON.
        output_path:   Where to write Stage 2 output JSON.
        top_k:         If set, keep only the top-K results.

    Returns:
        List of annotated company dicts sorted by stage2_score descending.
    """
    # ── Load stage 1 output ──────────────────────────────────────────────────
    with open(input_path) as f:
        stage1 = json.load(f)

    companies = stage1.get("companies", [])
    if not companies:
        print("Stage 2: no candidates from Stage 1 — nothing to rank.")
        _write(output_path, query_parsed, stage1, [], 0)
        return []

    # ── Build query string ───────────────────────────────────────────────────
    query_text   = build_query_text(query_parsed)
    query_tokens = tokenize(query_text)

    print(f"Stage 2: ranking {len(companies)} candidates …")
    print(f"  Query text: {query_text[:120]}")

    # ── BM25 ─────────────────────────────────────────────────────────────────
    bm25_index, _ = build_bm25_index(companies)
    b_scores      = bm25_scores(bm25_index, query_tokens)

    # ── Embeddings (optional) ─────────────────────────────────────────────────
    if USE_EMBEDDINGS and _EMBED_AVAILABLE:
        corpus_texts = [build_company_text(co) for co in companies]
        corpus_vecs  = embed_texts(corpus_texts)
        query_vec    = embed_texts([query_text])[0]
        e_scores     = cosine_sim_matrix(query_vec, corpus_vecs)
        # shift cosine from [-1,1] to [0,1]
        e_scores_norm = [(s + 1) / 2 for s in e_scores]
    else:
        e_scores_norm = [0.0] * len(companies)
        effective_alpha = 1.0
        effective_beta  = 0.0

    if USE_EMBEDDINGS and _EMBED_AVAILABLE:
        effective_alpha = ALPHA
        effective_beta  = BETA
    else:
        effective_alpha = 1.0
        effective_beta  = 0.0

    # ── Combine scores ────────────────────────────────────────────────────────
    scored = []
    for i, co in enumerate(companies):
        final = effective_alpha * b_scores[i] + effective_beta * e_scores_norm[i]
        scored.append({
            **co,
            "_retrieval": {
                "bm25_norm":   round(b_scores[i], 4),
                "embed_sim":   round(e_scores_norm[i], 4),
                "stage2_score": round(final, 4),
                "used_embeddings": USE_EMBEDDINGS and _EMBED_AVAILABLE,
            },
        })

    scored.sort(key=lambda x: x["_retrieval"]["stage2_score"], reverse=True)

    if top_k is not None:
        scored = scored[:top_k]

    # ── Write output ──────────────────────────────────────────────────────────
    _write(output_path, query_parsed, stage1, scored, len(companies))

    print(
        f"Stage 2: {len(companies)} → {len(scored)} candidates "
        f"(top_k={top_k}) → {output_path}"
    )
    return scored


def _write(
    output_path: str,
    query_parsed: dict,
    stage1: dict,
    companies: list[dict],
    total_input: int,
) -> None:
    output = {
        "query_parsed":   query_parsed,
        "total_input":    total_input,
        "total_output":   len(companies),
        "alpha":          ALPHA,
        "beta":           BETA,
        "used_embeddings": USE_EMBEDDINGS and _EMBED_AVAILABLE,
        "companies":      companies,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


# ── Quick self-test ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "German packaging suppliers for food and beverage"
    )

    # Simulate a minimal parsed query (without calling the LLM)
    dummy_parsed = {
        "original_query":    query,
        "structured_filters": {},
        "semantic_keywords": ["packaging", "supplier", "food", "beverage", "Germany"],
        "role_label":        "Supplier",
        "reasoning":         "Self-test",
    }

    results = run(dummy_parsed, top_k=50)
    for rank, co in enumerate(results, 1):
        r = co["_retrieval"]
        print(
            f"{rank:2}. [{r['stage2_score']:.3f}]  "
            f"{co.get('operational_name', 'N/A'):40s}  "
            f"{co.get('address_country_code', '').upper()}"
        )
