#!/usr/bin/env python3
"""
Intent Qualification System for Company Search.

Multi-stage pipeline (the "funnel"):
  Stage 1 – Query Deconstruction  : GPT-4o-mini parses the free-text query
                                     into a structured SearchSpec.
  Stage 2 – Hybrid Retrieval      : Attribute filtering + vector search
                                     narrows 477 companies → top 100.
  Stage 3 – Heuristic Scoring     : Fast code-based logic re-ranks top 100
                                     → top 20.
  Stage 4 – LLM Qualification     : GPT-4o evaluates the top 20 in a single
                                     batched prompt → final 5-10 results.

Usage:
    python main.py "Companies supplying packaging for D2C cosmetics brands"

Environment:
    OPENAI_API_KEY  – required (set in .env or shell environment)
"""

import ast
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE = os.path.join(os.path.dirname(__file__), "companies.jsonl")
EMBEDDING_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".embedding_cache.json")

TOP_RETRIEVAL = 100   # Stage 2 → Stage 3
TOP_HEURISTIC = 20    # Stage 3 → Stage 4
FINAL_RESULTS = 10    # Stage 4 → output

EMBEDDING_MODEL = "text-embedding-3-small"
DECONSTRUCT_MODEL = "gpt-4o-mini"
QUALIFY_MODEL = "gpt-4o"

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SearchSpec:
    """Structured output produced by Stage 1 (Query Deconstruction)."""

    locations: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    # e.g. ["Public", "Private"]
    company_types: list[str] = field(default_factory=list)
    # e.g. ["Supplier", "Competitor", "Customer"]
    roles: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    min_revenue: float | None = None
    max_revenue: float | None = None
    min_employees: int | None = None
    max_employees: int | None = None
    raw_query: str = ""


@dataclass
class ScoredCompany:
    """A company record paired with its pipeline score and LLM justification."""

    company: dict[str, Any]
    score: float = 0.0
    justification: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def _safe_parse(value: Any) -> dict:
    """Parse a field stored as a stringified Python dict (e.g. address, naics)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return {}


def _company_text(company: dict) -> str:
    """Build a single text blob suitable for embedding from a company record."""
    parts: list[str] = []
    if company.get("operational_name"):
        parts.append(company["operational_name"])
    if company.get("description"):
        parts.append(company["description"])
    if company.get("core_offerings"):
        parts.append(", ".join(company["core_offerings"]))
    if company.get("target_markets"):
        parts.append(", ".join(company["target_markets"]))
    naics = _safe_parse(company.get("primary_naics"))
    if naics.get("label"):
        parts.append(naics["label"])
    if company.get("business_model"):
        parts.append(", ".join(company["business_model"]))
    return " | ".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_companies(path: str = DATA_FILE) -> list[dict]:
    """Load company records from a JSONL file."""
    companies: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                companies.append(json.loads(line))
    return companies


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache (disk-backed, keyed by text content)
# ─────────────────────────────────────────────────────────────────────────────


def _load_embedding_cache() -> dict[str, list[float]]:
    if os.path.exists(EMBEDDING_CACHE_FILE):
        try:
            with open(EMBEDDING_CACHE_FILE, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _save_embedding_cache(cache: dict[str, list[float]]) -> None:
    with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)


def _embed(
    client: OpenAI,
    texts: list[str],
    cache: dict[str, list[float]] | None = None,
) -> list[list[float]]:
    """
    Embed a list of texts using the OpenAI Embeddings API.

    Results are stored in *cache* (mutated in-place) so repeated calls with
    the same text avoid a round-trip to the API.
    """
    if cache is None:
        cache = {}

    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        if text in cache:
            results[i] = cache[text]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Fetch uncached embeddings in chunks of 100 (well within API limits)
    chunk_size = 100
    new_embeddings: list[list[float]] = []
    for start in range(0, len(uncached_texts), chunk_size):
        chunk = uncached_texts[start : start + chunk_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        new_embeddings.extend(item.embedding for item in response.data)

    for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
        cache[text] = emb
        results[idx] = emb

    return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Query Deconstruction
# ─────────────────────────────────────────────────────────────────────────────

_DECONSTRUCT_SYSTEM = """
You are a query-parsing assistant for a B2B company-search engine.
Given a free-text business search query, extract a JSON object with these fields:
{
  "locations":      ["country or city names mentioned"],
  "industries":     ["industry or sector names"],
  "company_types":  ["Public" | "Private" | other descriptors],
  "roles":          ["Supplier" | "Provider" | "Competitor" | "Customer" | "Manufacturer" | "Distributor" | ...],
  "keywords":       ["important product / technology / domain keywords"],
  "min_revenue":    number | null,
  "max_revenue":    number | null,
  "min_employees":  integer | null,
  "max_employees":  integer | null
}
Return ONLY valid JSON. Do not include markdown fences or any other text.
""".strip()


def stage1_deconstruct(client: OpenAI, query: str) -> SearchSpec:
    """
    Stage 1 – Query Deconstruction.

    A lightweight LLM (GPT-4o-mini) parses the free-text query into a
    structured SearchSpec containing extracted entities, constraints, and roles.
    """
    response = client.chat.completions.create(
        model=DECONSTRUCT_MODEL,
        messages=[
            {"role": "system", "content": _DECONSTRUCT_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=512,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            logger.warning("Stage 1: could not parse LLM response as JSON: %r", raw)
            data = {}

    return SearchSpec(
        locations=data.get("locations", []),
        industries=data.get("industries", []),
        company_types=data.get("company_types", []),
        roles=data.get("roles", []),
        keywords=data.get("keywords", []),
        min_revenue=data.get("min_revenue"),
        max_revenue=data.get("max_revenue"),
        min_employees=data.get("min_employees"),
        max_employees=data.get("max_employees"),
        raw_query=query,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Hybrid Retrieval
# ─────────────────────────────────────────────────────────────────────────────


def _attribute_filter(company: dict, spec: SearchSpec) -> bool:
    """
    Hard-exclude a company that can *never* satisfy the search spec.

    Returns False only when a constraint is explicitly set AND clearly violated.
    Unknown / null field values are left in (handled with inference in Stage 3).
    """
    # Public / Private constraint
    if spec.company_types:
        types_lower = {t.lower() for t in spec.company_types}
        if "public" in types_lower and not company.get("is_public"):
            return False
        if "private" in types_lower and company.get("is_public"):
            return False

    # Revenue constraints (skip if revenue is unknown)
    rev = company.get("revenue")
    if rev is not None:
        if spec.min_revenue is not None and rev < spec.min_revenue:
            return False
        if spec.max_revenue is not None and rev > spec.max_revenue:
            return False

    # Employee count constraints (skip if unknown)
    emp = company.get("employee_count")
    if emp is not None:
        if spec.min_employees is not None and emp < spec.min_employees:
            return False
        if spec.max_employees is not None and emp > spec.max_employees:
            return False

    # Location filter – match against country_code, region, town, or website TLD
    if spec.locations:
        locations_lower = [loc.lower() for loc in spec.locations]
        addr = _safe_parse(company.get("address"))
        country_code = (addr.get("country_code") or "").lower()
        region = (addr.get("region_name") or "").lower()
        town = (addr.get("town") or "").lower()
        website = (company.get("website") or "").lower()
        # Infer region from website TLD when address is missing
        tld = website.rsplit(".", 1)[-1] if "." in website else ""
        location_fields = [country_code, region, town, tld]
        matched = any(
            any(loc in field or field in loc for field in location_fields if field)
            for loc in locations_lower
        )
        if not matched:
            return False

    return True


def stage2_retrieve(
    client: OpenAI,
    companies: list[dict],
    spec: SearchSpec,
    cache: dict[str, list[float]],
    top_n: int = TOP_RETRIEVAL,
) -> list[dict]:
    """
    Stage 2 – Hybrid Retrieval.

    1. Attribute filtering removes companies that cannot satisfy hard constraints.
    2. Vector search (text-embedding-3-small) scores the remainder by semantic
       similarity to the query and returns the top *top_n* candidates.

    Result: 477 companies → top 100 potential matches.
    """
    # ── Attribute filtering ──────────────────────────────────────────────────
    candidates = [c for c in companies if _attribute_filter(c, spec)]
    if not candidates:
        # Fallback: skip hard filter so we always have something to work with
        candidates = companies

    # ── Build query embedding text ───────────────────────────────────────────
    query_parts = [spec.raw_query]
    if spec.keywords:
        query_parts.append(" ".join(spec.keywords))
    if spec.industries:
        query_parts.append(" ".join(spec.industries))
    if spec.roles:
        query_parts.append(" ".join(spec.roles))
    query_text = " ".join(query_parts)

    # ── Embed query + all candidates ─────────────────────────────────────────
    texts = [query_text] + [_company_text(c) for c in candidates]
    embeddings = _embed(client, texts, cache)
    query_embedding = embeddings[0]
    company_embeddings = embeddings[1:]

    # ── Rank by cosine similarity ─────────────────────────────────────────────
    scored = [
        (c, _cosine_similarity(query_embedding, emb))
        for c, emb in zip(candidates, company_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_n]]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Heuristic Scoring & Re-ranking
# ─────────────────────────────────────────────────────────────────────────────

# Business models that suggest a company *is* a supplier / manufacturer
_SUPPLIER_BIZ_MODELS = {
    "manufacturing",
    "wholesale",
    "business-to-business",
    "distributor",
}
# Business models that suggest a company is *not* a B2B supplier
_CONSUMER_ONLY_BIZ_MODELS = {"business-to-consumer", "retail"}

_ROLE_POSITIVE_MODELS: dict[str, set[str]] = {
    "supplier": _SUPPLIER_BIZ_MODELS,
    "manufacturer": {"manufacturing"},
    "distributor": {"wholesale", "distributor"},
    "provider": {"service provider", "business-to-business", "saas"},
    "retailer": {"retail", "business-to-consumer"},
}


def _heuristic_score(company: dict, spec: SearchSpec) -> float:
    """
    Compute a fast heuristic score (0–1) for a single company.

    Applies keyword boosting, role alignment, negative constraints, and
    data-quality bonuses.  Missing fields are handled via inference logic.
    """
    score = 0.50  # Neutral baseline

    biz_models = {m.lower() for m in (company.get("business_model") or [])}
    description = (company.get("description") or "").lower()
    core_offerings_text = " ".join(
        o.lower() for o in (company.get("core_offerings") or [])
    )
    target_markets_text = " ".join(
        t.lower() for t in (company.get("target_markets") or [])
    )
    naics = _safe_parse(company.get("primary_naics"))
    naics_label = (naics.get("label") or "").lower()

    all_text = " ".join(
        [description, naics_label, core_offerings_text, target_markets_text]
    )

    # ── Role alignment ───────────────────────────────────────────────────────
    for role in (r.lower() for r in spec.roles):
        positive = _ROLE_POSITIVE_MODELS.get(role, set())
        if positive and biz_models & positive:
            score += 0.15

    # ── Keyword boosting ─────────────────────────────────────────────────────
    keyword_hits = sum(1 for kw in spec.keywords if kw.lower() in all_text)
    score += 0.05 * keyword_hits

    # ── Industry alignment ───────────────────────────────────────────────────
    industry_hits = sum(1 for ind in spec.industries if ind.lower() in all_text)
    score += 0.05 * industry_hits

    # ── Negative constraints ─────────────────────────────────────────────────
    # Penalise pure B2C companies when the user is looking for a B2B supplier
    wants_supplier = any(
        r.lower() in {"supplier", "manufacturer", "distributor"} for r in spec.roles
    )
    if wants_supplier and biz_models and biz_models.issubset(_CONSUMER_ONLY_BIZ_MODELS):
        score -= 0.30

    # ── Data quality & scale inference ───────────────────────────────────────
    if company.get("description"):
        score += 0.05
    if company.get("employee_count") is not None:
        score += 0.02
    if company.get("revenue") is not None:
        score += 0.02
    # Revenue missing but large employee count → likely mid-to-large company
    if company.get("revenue") is None and (company.get("employee_count") or 0) > 500:
        score += 0.03

    return max(0.0, min(1.0, score))


def stage3_heuristic(
    candidates: list[dict],
    spec: SearchSpec,
    top_n: int = TOP_HEURISTIC,
) -> list[ScoredCompany]:
    """
    Stage 3 – Heuristic Scoring & Re-ranking.

    Applies fast, code-based logic (role alignment, keyword boosting, negative
    constraints) to the top-100 candidates and returns the top *top_n*.
    """
    scored = [
        ScoredCompany(company=c, score=_heuristic_score(c, spec))
        for c in candidates
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – LLM Qualification
# ─────────────────────────────────────────────────────────────────────────────

_QUALIFY_SYSTEM = """
You are a senior business intelligence analyst.
You will receive a user's search intent and a numbered list of candidate companies.
For each company, evaluate whether it satisfies the user's intent and provide:
  - "confidence": a float between 0.0 (no match) and 1.0 (perfect match)
  - "justification": a single sentence explaining your rating

Respond ONLY with a JSON array of objects in the SAME ORDER as the candidates:
[
  {"confidence": 0.92, "justification": "..."},
  ...
]
Return ONLY valid JSON. Do not include markdown fences or any other text.
""".strip()


def _format_company_for_prompt(idx: int, sc: ScoredCompany) -> str:
    """Format one company for inclusion in the Stage-4 prompt."""
    c = sc.company
    naics = _safe_parse(c.get("primary_naics"))
    addr = _safe_parse(c.get("address"))
    revenue_str = (
        "${:,.0f}".format(c["revenue"]) if c.get("revenue") is not None else "N/A"
    )
    employees_str = (
        str(int(c["employee_count"])) if c.get("employee_count") is not None else "N/A"
    )
    # Limit core_offerings to the first 5 items to keep the prompt concise
    offerings = (c.get("core_offerings") or [])[:5]
    lines = [
        f"[{idx}] {c.get('operational_name', 'Unknown')} ({c.get('website', 'N/A')})",
        f"  NAICS: {naics.get('label', 'N/A')} ({naics.get('code', 'N/A')})",
        f"  Country: {addr.get('country_code', 'N/A').upper()}",
        f"  Business Model: {', '.join(c.get('business_model') or ['N/A'])}",
        f"  Description: {c.get('description') or 'N/A'}",
        f"  Core Offerings: {', '.join(offerings) if offerings else 'N/A'}",
        f"  Revenue: {revenue_str}",
        f"  Employees: {employees_str}",
        f"  Public: {c.get('is_public', 'N/A')}",
    ]
    return "\n".join(lines)


def stage4_qualify(
    client: OpenAI,
    candidates: list[ScoredCompany],
    spec: SearchSpec,
    final_n: int = FINAL_RESULTS,
) -> list[ScoredCompany]:
    """
    Stage 4 – LLM Qualification.

    Sends the top-20 candidates to GPT-4o in a single batched prompt.
    The LLM provides a confidence score (0-1) and a one-sentence justification
    for each candidate.  The final score blends LLM confidence (70%) with the
    heuristic score (30%) and returns the top *final_n* results.
    """
    companies_block = "\n\n".join(
        _format_company_for_prompt(i + 1, sc) for i, sc in enumerate(candidates)
    )
    user_message = (
        f"User Intent: {spec.raw_query}\n\n"
        f"Parsed Intent:\n"
        f"  Locations:     {spec.locations or 'any'}\n"
        f"  Industries:    {spec.industries or 'any'}\n"
        f"  Roles:         {spec.roles or 'any'}\n"
        f"  Keywords:      {spec.keywords or 'none'}\n"
        f"  Company Types: {spec.company_types or 'any'}\n"
        f"  Min Revenue:   {spec.min_revenue}\n"
        f"  Max Revenue:   {spec.max_revenue}\n"
        f"  Min Employees: {spec.min_employees}\n"
        f"  Max Employees: {spec.max_employees}\n\n"
        f"Candidates:\n{companies_block}"
    )

    response = client.chat.completions.create(
        model=QUALIFY_MODEL,
        messages=[
            {"role": "system", "content": _QUALIFY_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=1024,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        evaluations = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            evaluations = json.loads(match.group())
        else:
            logger.warning("Stage 4: could not parse LLM response as JSON: %r", raw)
            evaluations = []

    results: list[ScoredCompany] = []
    for i, sc in enumerate(candidates):
        if i < len(evaluations):
            ev = evaluations[i]
            llm_confidence = float(ev.get("confidence", 0.0))
            justification = ev.get("justification", "")
            # Blend: 70 % LLM confidence + 30 % heuristic score
            combined = 0.7 * llm_confidence + 0.3 * sc.score
        else:
            combined = sc.score
            justification = ""
        results.append(
            ScoredCompany(
                company=sc.company,
                score=combined,
                justification=justification,
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:final_n]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def search(
    query: str,
    data_file: str = DATA_FILE,
    top_retrieval: int = TOP_RETRIEVAL,
    top_heuristic: int = TOP_HEURISTIC,
    final_n: int = FINAL_RESULTS,
    verbose: bool = True,
) -> list[ScoredCompany]:
    """
    Run the full Intent Qualification pipeline for *query*.

    Parameters
    ----------
    query          : Free-text business search query.
    data_file      : Path to the JSONL company database.
    top_retrieval  : Number of candidates passed from Stage 2 to Stage 3.
    top_heuristic  : Number of candidates passed from Stage 3 to Stage 4.
    final_n        : Maximum number of results returned.
    verbose        : Print progress and results to stdout.

    Returns
    -------
    List of ScoredCompany objects sorted by descending score.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    client = OpenAI(api_key=api_key)
    cache = _load_embedding_cache()

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n{'=' * 60}")
    _log(f"Query: {query}")
    _log(f"{'=' * 60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    _log("Loading company data …")
    companies = load_companies(data_file)
    _log(f"  {len(companies)} companies loaded.\n")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    _log("Stage 1 – Query Deconstruction …")
    spec = stage1_deconstruct(client, query)
    _log(f"  Locations:     {spec.locations}")
    _log(f"  Industries:    {spec.industries}")
    _log(f"  Roles:         {spec.roles}")
    _log(f"  Keywords:      {spec.keywords}")
    _log(f"  Company Types: {spec.company_types}")
    _log(f"  Min Revenue:   {spec.min_revenue}")
    _log(f"  Min Employees: {spec.min_employees}\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    _log(f"Stage 2 – Hybrid Retrieval (→ top {top_retrieval}) …")
    top_candidates = stage2_retrieve(
        client, companies, spec, cache, top_n=top_retrieval
    )
    _log(f"  {len(top_candidates)} candidates retrieved.\n")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    _log(f"Stage 3 – Heuristic Scoring (→ top {top_heuristic}) …")
    heuristic_results = stage3_heuristic(top_candidates, spec, top_n=top_heuristic)
    _log(f"  Top {len(heuristic_results)} candidates after heuristic scoring.\n")

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    _log(f"Stage 4 – LLM Qualification (→ top {final_n}) …")
    final_results = stage4_qualify(client, heuristic_results, spec, final_n=final_n)
    _log(f"  Final {len(final_results)} results.\n")

    # Persist updated embedding cache
    _save_embedding_cache(cache)

    # ── Print results ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RESULTS for: '{query}'")
        print(f"{'=' * 60}")
        for i, result in enumerate(final_results):
            c = result.company
            naics = _safe_parse(c.get("primary_naics"))
            addr = _safe_parse(c.get("address"))
            print(
                f"\n#{i + 1}  {c.get('operational_name', 'Unknown')}"
                f"  (score: {result.score:.2f})"
            )
            print(f"     Website:       {c.get('website', 'N/A')}")
            print(f"     Industry:      {naics.get('label', 'N/A')}")
            print(f"     Country:       {addr.get('country_code', 'N/A').upper()}")
            if result.justification:
                print(f"     Justification: {result.justification}")
        print()

    return final_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default demo query from the problem statement
        _query = "Companies supplying packaging for D2C cosmetics brands"
    else:
        _query = " ".join(sys.argv[1:])

    search(_query)
