#!/usr/bin/env python3
"""
Intent Qualification System
============================
Ranks and qualifies companies against user queries using a multi-stage pipeline:

  1. Query Analysis   – extract structured constraints (country, size, public, year, etc.)
  2. Pre-Filter       – cheap hard-filter on structured fields
  3. Semantic Scoring – TF-IDF cosine similarity on rich company text profiles
  4. LLM Re-Ranking   – optional OpenAI call for top-K candidates on complex queries

Usage:
    python solution.py                      # run all 12 benchmark queries
    python solution.py --query "Fintech companies in Europe"
    python solution.py --top 10             # show top-10 per query (default: 15)
    python solution.py --llm                # enable LLM re-ranking (requires OPENAI_API_KEY)
    python solution.py --data path/to/companies.jsonl
"""

import argparse
import ast
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    "Logistic companies in Romania",
    "Public software companies with more than 1,000 employees",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Construction companies in the United States with revenue over $50 million",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "E-commerce companies using Shopify or similar platforms",
    "Renewable energy equipment manufacturers in Scandinavia",
    "Companies that manufacture or supply critical components for electric vehicle battery production",
]

# ---------------------------------------------------------------------------
# Geography helpers
# ---------------------------------------------------------------------------
COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "romania": "ro",
    "france": "fr",
    "germany": "de",
    "switzerland": "ch",
    "united states": "us",
    "usa": "us",
    "uk": "gb",
    "united kingdom": "gb",
    "great britain": "gb",
    "sweden": "se",
    "norway": "no",
    "denmark": "dk",
    "finland": "fi",
    "netherlands": "nl",
    "poland": "pl",
    "italy": "it",
    "spain": "es",
    "austria": "at",
    "belgium": "be",
    "portugal": "pt",
    "czech republic": "cz",
    "hungary": "hu",
    "bulgaria": "bg",
    "croatia": "hr",
    "serbia": "rs",
    "slovakia": "sk",
    "slovenia": "si",
    "latvia": "lv",
    "lithuania": "lt",
    "estonia": "ee",
    "luxembourg": "lu",
    "malta": "mt",
    "cyprus": "cy",
    "greece": "gr",
    "ireland": "ie",
    "iceland": "is",
    "china": "cn",
    "india": "in",
    "japan": "jp",
    "canada": "ca",
    "australia": "au",
    "singapore": "sg",
    "brazil": "br",
    "south korea": "kr",
    "new zealand": "nz",
    "indonesia": "id",
}

EUROPE_CODES: set[str] = {
    "ro", "fr", "de", "ch", "gb", "se", "no", "dk", "fi", "nl", "pl", "it",
    "es", "at", "be", "pt", "cz", "hu", "bg", "hr", "rs", "sk", "si", "lv",
    "lt", "ee", "lu", "mt", "cy", "gr", "ie", "is",
}

SCANDINAVIA_CODES: set[str] = {"se", "no", "dk", "fi"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Company:
    """Parsed company record with typed fields."""

    raw: dict[str, Any]
    website: str = ""
    operational_name: str = ""
    year_founded: Optional[float] = None
    country_code: Optional[str] = None
    region_name: Optional[str] = None
    town: Optional[str] = None
    employee_count: Optional[float] = None
    revenue: Optional[float] = None
    primary_naics_code: Optional[str] = None
    primary_naics_label: str = ""
    secondary_naics_labels: list[str] = field(default_factory=list)
    description: str = ""
    business_model: list[str] = field(default_factory=list)
    target_markets: list[str] = field(default_factory=list)
    core_offerings: list[str] = field(default_factory=list)
    is_public: bool = False
    text_profile: str = ""  # pre-built flat text for TF-IDF


@dataclass
class QueryConstraints:
    """Parsed constraints extracted from a free-text query."""

    country_codes: set[str] = field(default_factory=set)  # exact match
    require_europe: bool = False
    require_scandinavia: bool = False
    require_public: Optional[bool] = None
    min_employees: Optional[int] = None
    max_employees: Optional[int] = None
    min_revenue: Optional[float] = None
    max_revenue: Optional[float] = None
    min_year_founded: Optional[int] = None
    max_year_founded: Optional[int] = None
    required_business_models: list[str] = field(default_factory=list)
    semantic_query: str = ""  # cleaned query text for TF-IDF


# ---------------------------------------------------------------------------
# Company loader & parser
# ---------------------------------------------------------------------------
def _extract_country_code(address_str: str) -> Optional[str]:
    """Robustly extract country_code from the address string."""
    if not address_str:
        return None
    m = re.search(r"'country_code'\s*:\s*'([^']+)'", address_str)
    if m:
        return m.group(1).lower()
    m = re.search(r'"country_code"\s*:\s*"([^"]+)"', address_str)
    if m:
        return m.group(1).lower()
    return None


def _extract_region(address_str: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (region_name, town) from the address string."""
    region = None
    town = None
    m = re.search(r"'region_name'\s*:\s*'([^']+)'", address_str)
    if m:
        region = m.group(1)
    m = re.search(r"'town'\s*:\s*'([^']+)'", address_str)
    if m:
        town = m.group(1)
    return region, town


def _parse_naics(naics_val: Any) -> tuple[Optional[str], str]:
    """Parse a NAICS field that may be a dict or a string-encoded dict."""
    if not naics_val:
        return None, ""
    if isinstance(naics_val, dict):
        return naics_val.get("code"), naics_val.get("label", "")
    if isinstance(naics_val, str):
        try:
            d = ast.literal_eval(naics_val)
            return d.get("code"), d.get("label", "")
        except (ValueError, SyntaxError):
            pass
        m = re.search(r"'label'\s*:\s*'([^']+)'", naics_val)
        if m:
            return None, m.group(1)
        m = re.search(r"'code'\s*:\s*'([^']+)'", naics_val)
        if m:
            return m.group(1), ""
    return None, ""


def _parse_secondary_naics(val: Any) -> list[str]:
    labels = []
    if not val:
        return labels
    if isinstance(val, list):
        for item in val:
            _, label = _parse_naics(item)
            if label:
                labels.append(label)
    elif isinstance(val, str):
        for m in re.finditer(r"'label'\s*:\s*'([^']+)'", val):
            labels.append(m.group(1))
    return labels


def _build_text_profile(c: Company) -> str:
    """Build a single flat text string from all useful fields for TF-IDF."""
    parts: list[str] = []

    if c.operational_name:
        parts.append(c.operational_name)
    if c.website:
        parts.append(c.website)
    if c.primary_naics_label:
        # Repeat to up-weight industry label
        parts.extend([c.primary_naics_label] * 3)
    if c.secondary_naics_labels:
        parts.extend(c.secondary_naics_labels * 2)
    if c.description:
        parts.append(c.description)
    if c.business_model:
        parts.extend(c.business_model * 2)
    if c.target_markets:
        parts.extend(c.target_markets * 2)
    if c.core_offerings:
        parts.extend(c.core_offerings * 2)

    # Append geography tokens so that city/region mentions help
    if c.country_code:
        parts.append(c.country_code)
    if c.region_name:
        parts.append(c.region_name)
    if c.town:
        parts.append(c.town)

    return " ".join(parts)


def load_companies(path: str) -> list[Company]:
    """Load and parse companies from a JSONL file."""
    companies: list[Company] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            c = Company(raw=raw)
            c.website = raw.get("website") or ""
            c.operational_name = raw.get("operational_name") or ""
            c.year_founded = raw.get("year_founded")
            c.employee_count = raw.get("employee_count")
            c.revenue = raw.get("revenue")
            c.is_public = bool(raw.get("is_public"))
            c.description = raw.get("description") or ""
            c.business_model = raw.get("business_model") or []
            c.target_markets = raw.get("target_markets") or []
            c.core_offerings = raw.get("core_offerings") or []

            # Address
            addr_str = raw.get("address") or ""
            if isinstance(addr_str, dict):
                addr_str = str(addr_str)
            c.country_code = _extract_country_code(addr_str)
            c.region_name, c.town = _extract_region(addr_str)

            # NAICS
            c.primary_naics_code, c.primary_naics_label = _parse_naics(raw.get("primary_naics"))
            c.secondary_naics_labels = _parse_secondary_naics(raw.get("secondary_naics"))

            c.text_profile = _build_text_profile(c)
            companies.append(c)

    return companies


# ---------------------------------------------------------------------------
# Query analyser
# ---------------------------------------------------------------------------
_MILLION = 1_000_000
_BILLION = 1_000_000_000

_NUM_WORDS: dict[str, float] = {
    "thousand": 1_000,
    "million": _MILLION,
    "billion": _BILLION,
}


def _parse_number(text: str) -> Optional[float]:
    """Parse a number like '50 million', '1,000', '1000' from a text fragment."""
    text = text.strip().replace(",", "")
    for word, multiplier in _NUM_WORDS.items():
        if word in text.lower():
            m = re.search(r"[\d.]+", text)
            if m:
                return float(m.group()) * multiplier
    m = re.search(r"[\d.]+", text)
    if m:
        return float(m.group())
    return None


def parse_query(query: str) -> QueryConstraints:
    """Extract structured constraints and a cleaned semantic query."""
    qc = QueryConstraints()
    q_lower = query.lower()

    # ---- Public / private ----
    if re.search(r"\bpublic\b", q_lower):
        qc.require_public = True
    if re.search(r"\bprivate\b", q_lower):
        qc.require_public = False

    # ---- Geography ----
    # Check for Europe / Scandinavia first (regions, not countries)
    if re.search(r"\beurope\b|\beuropean\b", q_lower):
        qc.require_europe = True
    if re.search(r"\bscandinavia\b|\bscandinavian\b|\bnordic\b", q_lower):
        qc.require_scandinavia = True

    # Multi-word country names first, then single-word
    for name, code in sorted(COUNTRY_NAME_TO_CODE.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(name) + r"\b", q_lower):
            qc.country_codes.add(code)

    # ---- Employee count ----
    # "more than 1,000 employees" / "fewer than 200 employees"
    m = re.search(
        r"(?:more than|over|greater than|at least)\s+([\d,]+(?:\s+(?:thousand|million))?)\s+employees?",
        q_lower,
    )
    if m:
        qc.min_employees = int(_parse_number(m.group(1)) or 0)
    m = re.search(
        r"(?:fewer than|less than|under|below)\s+([\d,]+(?:\s+(?:thousand|million))?)\s+employees?",
        q_lower,
    )
    if m:
        qc.max_employees = int(_parse_number(m.group(1)) or 0)

    # ---- Revenue ----
    m = re.search(
        r"revenue\s+(?:over|above|more than|greater than|exceeding)\s+\$?([\d,.]+\s*(?:million|billion|thousand)?)",
        q_lower,
    )
    if m:
        qc.min_revenue = _parse_number(m.group(1))
    m = re.search(
        r"revenue\s+(?:under|below|less than|fewer than)\s+\$?([\d,.]+\s*(?:million|billion|thousand)?)",
        q_lower,
    )
    if m:
        qc.max_revenue = _parse_number(m.group(1))

    # Alternate: "$50 million" pattern near revenue
    if qc.min_revenue is None:
        m = re.search(r"\$\s*([\d,.]+)\s*(million|billion|thousand)\b", q_lower)
        if m:
            base = float(m.group(1).replace(",", ""))
            multiplier = _NUM_WORDS.get(m.group(2), 1)
            val = base * multiplier
            # Determine direction from surrounding text
            surrounding = q_lower[max(0, m.start() - 30): m.end() + 30]
            if re.search(r"over|above|more than|greater|exceeding", surrounding):
                qc.min_revenue = val
            elif re.search(r"under|below|less than|fewer", surrounding):
                qc.max_revenue = val
            else:
                qc.min_revenue = val  # default to minimum threshold

    # ---- Year founded ----
    m = re.search(
        r"(?:founded|established|started)\s+(?:after|since)\s+(20\d\d|19\d\d)",
        q_lower,
    )
    if m:
        qc.min_year_founded = int(m.group(1))
    m = re.search(
        r"(?:founded|established|started)\s+(?:before|prior to|until)\s+(20\d\d|19\d\d)",
        q_lower,
    )
    if m:
        qc.max_year_founded = int(m.group(1))
    # Standalone "after 2018"
    if qc.min_year_founded is None:
        m = re.search(r"\bafter\s+(20\d\d|19\d\d)\b", q_lower)
        if m:
            qc.min_year_founded = int(m.group(1))

    # ---- Business model ----
    if re.search(r"\bb2b\b", q_lower):
        qc.required_business_models.append("B2B")
    if re.search(r"\bb2c\b", q_lower):
        qc.required_business_models.append("B2C")
    if re.search(r"\bsaas\b", q_lower):
        qc.required_business_models.append("SaaS")

    # ---- Build semantic query (feed everything to TF-IDF) ----
    # Remove geographic terms and numeric thresholds that are already handled by
    # hard filters so that TF-IDF focuses purely on industry / intent signals.
    semantic = query
    # Strip country/region mentions
    for name in sorted(COUNTRY_NAME_TO_CODE.keys(), key=len, reverse=True):
        semantic = re.sub(r"\b" + re.escape(name) + r"\b", " ", semantic, flags=re.IGNORECASE)
    for phrase in ("in europe", "european", "in scandinavia", "scandinavian", "nordic"):
        semantic = re.sub(re.escape(phrase), " ", semantic, flags=re.IGNORECASE)
    # Strip numeric thresholds (already applied as filters)
    semantic = re.sub(
        r"(?:more than|fewer than|over|under|less than|greater than|at least|below|above)"
        r"\s+[\d,]+(?:\s+(?:thousand|million|billion))?\s+(?:employees?|staff|revenue|workers?)",
        " ",
        semantic,
        flags=re.IGNORECASE,
    )
    semantic = re.sub(
        r"\$[\d,.]+\s*(?:million|billion|thousand)?\b",
        " ",
        semantic,
        flags=re.IGNORECASE,
    )
    semantic = re.sub(
        r"\b(?:founded|established)\s+(?:after|before|since)\s+\d{4}\b",
        " ",
        semantic,
        flags=re.IGNORECASE,
    )
    semantic = re.sub(r"\s{2,}", " ", semantic).strip()
    qc.semantic_query = semantic or query

    return qc


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def passes_filters(company: Company, qc: QueryConstraints) -> bool:
    """Return True if the company satisfies all hard structured constraints."""

    # Geography
    if qc.country_codes:
        if company.country_code not in qc.country_codes:
            return False

    if qc.require_europe and not qc.country_codes:
        if company.country_code not in EUROPE_CODES:
            return False

    if qc.require_scandinavia and not qc.country_codes:
        if company.country_code not in SCANDINAVIA_CODES:
            return False

    # Public
    if qc.require_public is not None:
        if company.is_public != qc.require_public:
            return False

    # Employees
    if qc.min_employees is not None:
        if company.employee_count is None or company.employee_count < qc.min_employees:
            return False
    if qc.max_employees is not None:
        if company.employee_count is None or company.employee_count > qc.max_employees:
            return False

    # Revenue
    if qc.min_revenue is not None:
        if company.revenue is None or company.revenue < qc.min_revenue:
            return False
    if qc.max_revenue is not None:
        if company.revenue is None or company.revenue > qc.max_revenue:
            return False

    # Year founded
    if qc.min_year_founded is not None:
        if company.year_founded is None or company.year_founded < qc.min_year_founded:
            return False
    if qc.max_year_founded is not None:
        if company.year_founded is None or company.year_founded > qc.max_year_founded:
            return False

    return True


# ---------------------------------------------------------------------------
# Semantic scoring (TF-IDF)
# ---------------------------------------------------------------------------
def build_tfidf_index(companies: list[Company]) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on all company text profiles."""
    stemmer = PorterStemmer()

    # Custom stop words: vague terms that appear across many profiles and tend
    # to cause false positives when they also appear incidentally in a query.
    extra_stops = {
        "company", "companies", "business", "businesses", "service", "services",
        "provider", "providers", "solution", "solutions", "product", "products",
        "global", "international", "world", "worldwide", "leading", "innovative",
        "advanced", "growing", "fast", "traditional", "new", "based", "provides",
        "offering", "including", "well", "also", "using", "used", "various",
        "large", "small", "medium", "major", "key", "primary", "secondary",
    }
    extra_stops_stemmed = {stemmer.stem(w) for w in extra_stops}
    # Common English stop tokens (abbreviated set to avoid sklearn dependency)
    english_stops = {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "that", "this", "it", "its",
        "their", "they", "we", "us", "our", "which", "who", "what", "how",
        "when", "where", "not", "no", "nor", "but", "if", "then", "than",
        "both", "each", "more", "most", "other", "such", "own", "same",
        "so", "can", "into", "through", "during", "about", "above", "between",
        "after", "before", "while", "although", "because", "since", "until",
    }

    def stemming_tokenizer(text: str) -> list[str]:
        tokens = re.findall(r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b", text.lower())
        stemmed = [stemmer.stem(t) for t in tokens if t not in english_stops]
        # Include unigrams and bigrams
        unigrams = stemmed
        bigrams = [f"{stemmed[i]} {stemmed[i+1]}" for i in range(len(stemmed) - 1)]
        return unigrams + bigrams

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        sublinear_tf=True,
        min_df=1,
        analyzer=stemming_tokenizer,
    )
    corpus = [c.text_profile for c in companies]
    vectorizer.fit(corpus)
    vectorizer._extra_stops = extra_stops_stemmed  # type: ignore[attr-defined]
    return vectorizer


def semantic_score(
    query: str,
    companies: list[Company],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """Return cosine similarity scores for each company against the query."""
    corpus = [c.text_profile for c in companies]
    company_vecs = vectorizer.transform(corpus)
    query_vec = vectorizer.transform([query])

    # Zero out extra stop words (stemmed) in the query vector so they don't
    # cause false positive matches on vague terms like "traditional", "growing".
    extra_stops: set[str] = getattr(vectorizer, "_extra_stops", set())
    if extra_stops:
        feature_names = vectorizer.get_feature_names_out()
        stop_indices = [
            i for i, tok in enumerate(feature_names) if tok in extra_stops
        ]
        if stop_indices:
            query_arr = query_vec.toarray()
            query_arr[:, stop_indices] = 0.0
            query_vec = csr_matrix(query_arr)

    scores = cosine_similarity(query_vec, company_vecs).flatten()
    return scores


# ---------------------------------------------------------------------------
# Optional LLM re-ranking (OpenAI)
# ---------------------------------------------------------------------------
def llm_qualify(
    query: str,
    candidates: list[tuple[Company, float]],
    top_k: int = 20,
) -> list[tuple[Company, float]]:
    """
    Re-rank the top `top_k` candidates using an LLM.

    Requires the OPENAI_API_KEY environment variable to be set.
    Falls back gracefully if the library is not installed or the key is missing.
    Returns the original list (possibly reordered) with updated scores.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return candidates

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("  [LLM] openai package not installed – skipping LLM re-ranking.")
        return candidates

    client = OpenAI(api_key=api_key)

    # Only send the top-K candidates to the LLM
    to_rank = candidates[:top_k]
    rest = candidates[top_k:]

    # Build a compact profile for each candidate
    def compact_profile(company: Company) -> str:
        parts = []
        if company.operational_name:
            parts.append(f"Name: {company.operational_name}")
        if company.primary_naics_label:
            parts.append(f"Industry: {company.primary_naics_label}")
        if company.description:
            parts.append(f"Description: {company.description[:300]}")
        if company.country_code:
            parts.append(f"Country: {company.country_code.upper()}")
        if company.core_offerings:
            parts.append(f"Core offerings: {', '.join(company.core_offerings[:5])}")
        return " | ".join(parts)

    companies_text = "\n".join(
        f"{i + 1}. {compact_profile(c)}" for i, (c, _) in enumerate(to_rank)
    )

    prompt = textwrap.dedent(f"""
        You are an expert business analyst.

        User query: "{query}"

        Below are {len(to_rank)} company candidates. For each one, score it 0–10
        based on how well it satisfies the user query (10 = perfect match, 0 = no match).
        Reply ONLY with a JSON array of integers in the same order as the list.
        Example: [8, 3, 10, 0, 7, ...]

        Companies:
        {companies_text}
    """).strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or "[]"
        # Extract JSON array
        m = re.search(r"\[[\d\s,]+\]", content)
        if not m:
            return candidates
        scores_raw = json.loads(m.group())
        if len(scores_raw) != len(to_rank):
            return candidates

        # Normalise to [0,1] and replace tfidf scores
        reranked = [
            (company, float(score) / 10.0)
            for (company, _), score in zip(to_rank, scores_raw)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked + rest

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError) as exc:
        print(f"  [LLM] Error parsing response: {exc}")
        return candidates
    except OSError as exc:
        print(f"  [LLM] Network/IO error during re-ranking: {exc}")
        return candidates


# ---------------------------------------------------------------------------
# Main qualification pipeline
# ---------------------------------------------------------------------------
def qualify(
    query: str,
    companies: list[Company],
    vectorizer: TfidfVectorizer,
    top_n: int = 15,
    use_llm: bool = False,
) -> list[tuple[Company, float]]:
    """
    Run the full qualification pipeline and return ranked (company, score) pairs.

    Stage 1 – Parse query into structured constraints.
    Stage 2 – Hard-filter companies that cannot possibly match.
    Stage 3 – Score surviving candidates with TF-IDF cosine similarity.
    Stage 4 – (Optional) Re-rank top-K with an LLM.
    """
    # Stage 1: parse
    qc = parse_query(query)

    # Stage 2: filter
    candidates = [c for c in companies if passes_filters(c, qc)]

    if not candidates:
        # Relax to full set if filters were too aggressive (e.g. no country data)
        candidates = companies

    # Stage 3: semantic scoring
    scores = semantic_score(qc.semantic_query, candidates, vectorizer)

    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)

    # Apply a soft business-model boost for queries that mention B2B/SaaS/etc.
    if qc.required_business_models:
        boosted = []
        for company, score in ranked:
            bm_lower = {bm.lower() for bm in company.business_model}
            required_lower = {r.lower() for r in qc.required_business_models}
            overlap = len(bm_lower & required_lower)
            boost = 1.0 + 0.15 * overlap
            boosted.append((company, score * boost))
        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)

    # Deduplicate by website (keep highest-scoring entry for each domain)
    seen_websites: set[str] = set()
    deduped: list[tuple[Company, float]] = []
    for company, score in ranked:
        key = company.website.lower().strip() if company.website else company.operational_name.lower()
        if key not in seen_websites:
            seen_websites.add(key)
            deduped.append((company, score))
    ranked = deduped

    # Filter out zero-score results (no meaningful semantic overlap)
    min_score_threshold = 1e-6
    ranked = [(c, s) for c, s in ranked if s >= min_score_threshold] or ranked[:top_n]

    # Stage 4: optional LLM re-ranking
    if use_llm:
        ranked = llm_qualify(query, ranked, top_k=20)

    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_result(idx: int, company: Company, score: float) -> str:
    """Format a single result line for console output."""
    name = company.operational_name or company.website or "Unknown"
    country = (company.country_code or "??").upper()
    industry = company.primary_naics_label or "N/A"
    pub = "PUBLIC" if company.is_public else "private"
    emp = f"{int(company.employee_count):,}" if company.employee_count else "N/A"
    rev = (
        f"${company.revenue / _MILLION:.0f}M"
        if company.revenue
        else "N/A"
    )
    return (
        f"  {idx:>2}. [{score:.3f}] {name:<35} "
        f"| {country} | {pub} | emp: {emp} | rev: {rev}\n"
        f"       Industry: {industry}"
    )


def print_results(query: str, results: list[tuple[Company, float]]) -> None:
    """Print formatted results for a query."""
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"Query: {query}")
    print(sep)
    if not results:
        print("  No matching companies found.")
        return
    for i, (company, score) in enumerate(results, 1):
        print(format_result(i, company, score))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Intent Qualification System")
    parser.add_argument(
        "--data",
        default="companies.jsonl",
        help="Path to the companies JSONL file (default: companies.jsonl)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query to run (runs all benchmark queries if not specified)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top results to display per query (default: 15)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM re-ranking (requires OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of formatted text",
    )
    args = parser.parse_args()

    print(f"Loading companies from {args.data}...")
    companies = load_companies(args.data)
    print(f"Loaded {len(companies)} companies.")

    print("Building TF-IDF index...")
    vectorizer = build_tfidf_index(companies)
    print("Index ready.\n")

    queries = [args.query] if args.query else BENCHMARK_QUERIES

    all_results: dict[str, list[dict]] = {}

    for query in queries:
        results = qualify(
            query=query,
            companies=companies,
            vectorizer=vectorizer,
            top_n=args.top,
            use_llm=args.llm,
        )

        if args.json_output:
            all_results[query] = [
                {
                    "rank": i + 1,
                    "score": round(score, 4),
                    "website": c.website,
                    "operational_name": c.operational_name,
                    "country_code": c.country_code,
                    "industry": c.primary_naics_label,
                    "is_public": c.is_public,
                    "employee_count": c.employee_count,
                    "revenue": c.revenue,
                }
                for i, (c, score) in enumerate(results)
            ]
        else:
            print_results(query, results)

    if args.json_output:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
