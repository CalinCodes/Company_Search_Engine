"""
Intent Qualification System
============================
A multi-stage pipeline that ranks and qualifies companies against user queries.

Architecture:
  1. QueryAnalyzer   – extracts structured constraints and semantic keywords from the query
  2. StructuredFilter – eliminates companies that violate hard constraints (country, size, etc.)
  3. SemanticScorer  – scores remaining companies with keyword-based relevance
  4. LLMReranker     – optional OpenAI reranker for complex queries (requires OPENAI_API_KEY)
  5. QualificationSystem – orchestrates the pipeline

Run:
    python solution.py                          # process all 12 built-in queries
    python solution.py --query "Pharma in CH"   # single query
    python solution.py --use-llm                # enable LLM reranking (needs OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Geography helpers
# ---------------------------------------------------------------------------

COUNTRY_MAP: dict[str, str] = {
    "afghanistan": "af", "albania": "al", "algeria": "dz", "andorra": "ad",
    "angola": "ao", "argentina": "ar", "armenia": "am", "australia": "au",
    "austria": "at", "azerbaijani": "az", "azerbaijan": "az",
    "bangladesh": "bd", "belarus": "by", "belgian": "be", "belgium": "be",
    "bolivia": "bo", "bosnia": "ba", "brazil": "br", "brazilian": "br",
    "bulgaria": "bg", "bulgarian": "bg",
    "cambodia": "kh", "canada": "ca", "canadian": "ca", "chile": "cl",
    "chinese": "cn", "china": "cn", "colombia": "co", "croatia": "hr",
    "croatian": "hr", "cyprus": "cy", "czech": "cz", "czech republic": "cz",
    "danish": "dk", "denmark": "dk",
    "egypt": "eg", "estonian": "ee", "estonia": "ee",
    "finland": "fi", "finnish": "fi", "france": "fr", "french": "fr",
    "georgian": "ge", "georgia": "ge", "german": "de", "germany": "de",
    "ghana": "gh", "greek": "gr", "greece": "gr",
    "hong kong": "hk", "hungary": "hu", "hungarian": "hu",
    "icelandic": "is", "iceland": "is", "india": "in", "indian": "in",
    "indonesia": "id", "irish": "ie", "ireland": "ie",
    "israel": "il", "israeli": "il", "italy": "it", "italian": "it",
    "japan": "jp", "japanese": "jp", "jordan": "jo",
    "kazakhstan": "kz", "kenya": "ke", "south korea": "kr", "korean": "kr",
    "latvia": "lv", "latvian": "lv", "lebanon": "lb",
    "lithuania": "lt", "lithuanian": "lt", "luxembourg": "lu",
    "malaysia": "my", "malta": "mt", "mexico": "mx", "mexican": "mx",
    "moldova": "md", "moroccan": "ma", "morocco": "ma",
    "netherlands": "nl", "dutch": "nl", "new zealand": "nz",
    "nigeria": "ng", "norwegian": "no", "norway": "no",
    "pakistan": "pk", "peru": "pe", "philippines": "ph", "polish": "pl",
    "poland": "pl", "portugal": "pt", "portuguese": "pt",
    "romanian": "ro", "romania": "ro", "russia": "ru", "russian": "ru",
    "saudi arabia": "sa", "serbian": "rs", "serbia": "rs",
    "singapore": "sg", "singaporean": "sg", "slovakia": "sk", "slovak": "sk",
    "slovenian": "si", "slovenia": "si", "south africa": "za",
    "spain": "es", "spanish": "es", "swedish": "se", "sweden": "se",
    "swiss": "ch", "switzerland": "ch",
    "taiwan": "tw", "thailand": "th", "turkey": "tr", "turkish": "tr",
    "ukraine": "ua", "ukrainian": "ua",
    "u.k.": "gb", "uk": "gb", "britain": "gb", "british": "gb",
    "united kingdom": "gb",
    "u.s.": "us", "usa": "us", "america": "us", "american": "us",
    "united states": "us",
    "uzbekistan": "uz", "vietnam": "vn",
}

EUROPE_CODES: frozenset[str] = frozenset({
    "ad", "al", "am", "at", "az", "ba", "be", "bg", "by", "ch", "cy",
    "cz", "de", "dk", "ee", "es", "fi", "fr", "gb", "ge", "gr", "hr",
    "hu", "ie", "il", "is", "it", "li", "lt", "lu", "lv", "mc", "md",
    "me", "mk", "mt", "nl", "no", "pl", "pt", "ro", "rs", "ru", "se",
    "si", "sk", "sm", "tr", "ua", "xk",
})

SCANDINAVIA_CODES: frozenset[str] = frozenset({"dk", "fi", "is", "no", "se"})

REGION_MAP: dict[str, frozenset[str]] = {
    "europe": EUROPE_CODES,
    "european": EUROPE_CODES,
    "scandinavia": SCANDINAVIA_CODES,
    "scandinavian": SCANDINAVIA_CODES,
    "nordic": SCANDINAVIA_CODES,
}

# ---------------------------------------------------------------------------
# Industry keyword groups (for scoring)
# ---------------------------------------------------------------------------

INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "logistics": [
        "logistics", "freight", "shipping", "transport", "warehousing",
        "warehouse", "supply chain", "cargo", "courier", "delivery",
        "fulfillment", "fleet", "trucking", "distribution", "forwarding",
        "customs brokerage", "last mile", "haulage", "freight forwarder",
        "3pl", "third-party logistics",
    ],
    "software": [
        "software", "saas", "platform", "application", "digital",
        "cloud", "technology", "tech", "it services", "developer",
        "programming", "computing", "artificial intelligence",
        "machine learning", "cybersecurity", "enterprise software",
        "information technology", "software development",
    ],
    "food_beverage": [
        "food", "beverage", "drink", "nutrition", "dairy", "meat", "bakery",
        "confectionery", "snack", "grocery", "agri-food", "agrifood",
        "catering", "brewery", "winery", "spirits", "beer",
        "wine", "coffee", "tea", "chocolate", "sugar", "flour",
        "food manufacturing", "food production", "food processing",
    ],
    "packaging": [
        "packaging", "container", "bottle", "label", "box", "carton",
        "wrapper", "pouch", "film", "corrugated", "folding carton",
        "glass container", "plastic packaging", "flexible packaging",
        "rigid packaging", "metal can", "paper packaging",
        "cosmetic packaging", "packaging materials", "packaging supplier",
    ],
    "construction": [
        "construction", "building contractor", "general contractor",
        "infrastructure", "civil engineering", "architecture",
        "real estate development", "structural", "renovation",
        "hvac", "plumbing", "electrical contractor", "road construction",
        "bridge construction", "commercial construction", "contractor",
    ],
    "pharmaceutical": [
        "pharmaceutical", "pharma", "drug", "medicine",
        "biotech", "biotechnology", "life science", "clinical", "vaccine",
        "therapeutic", "api", "generic drug", "specialty pharma",
        "biopharmaceutical", "drug discovery", "clinical trials",
        "contract manufacturing organization", "cmo",
    ],
    "hr": [
        "human resources", "recruitment", "talent management", "payroll",
        "workforce management", "hiring platform", "staffing",
        "hr software", "hris", "applicant tracking", "onboarding platform",
        "people management", "hr solutions", "hr platform",
    ],
    "clean_energy": [
        "clean energy", "renewable energy", "solar energy", "wind energy",
        "green energy", "sustainability", "carbon neutral", "net zero",
        "energy transition", "cleantech", "decarbonization", "bioenergy",
        "hydropower", "geothermal", "clean power", "green power",
        "wind power", "solar power",
    ],
    "fintech": [
        "fintech", "financial technology", "payments", "digital banking",
        "neobank", "digital payments", "lending platform", "insurtech",
        "cryptocurrency", "blockchain", "digital wallet", "remittance",
        "wealth management", "open banking", "payment processing",
        "financial services platform", "banking platform",
    ],
    "ecommerce": [
        "e-commerce", "ecommerce", "online retail", "shopify",
        "direct-to-consumer", "online store", "omnichannel retail",
        "online marketplace", "digital commerce", "retail technology",
        "d2c brand", "dtc",
    ],
    "renewable_equipment": [
        "wind turbine", "solar panel", "photovoltaic", "solar module",
        "inverter", "energy storage system", "battery storage",
        "offshore wind", "onshore wind", "renewable equipment",
        "wind rotor", "wind blade", "solar inverter", "pv module",
        "turbine manufacturer", "wind farm equipment",
    ],
    "ev_battery": [
        "electric vehicle battery", "ev battery", "lithium-ion battery",
        "lithium battery", "cathode material", "anode material",
        "electrolyte", "battery cell", "battery pack",
        "battery management system", "nickel sulfate", "cobalt sulfate",
        "manganese", "graphite anode", "battery separator",
        "pouch cell", "cylindrical cell", "prismatic cell",
        "solid-state battery", "battery components", "battery materials",
    ],
    "cosmetics": [
        "cosmetic", "beauty", "skincare", "makeup", "personal care",
        "fragrance", "perfume", "haircare", "toiletry",
    ],
}

# NAICS code prefixes that map to each industry group.
# A company with a matching primary NAICS code receives a strong relevance boost.
NAICS_PREFIXES: dict[str, list[str]] = {
    "logistics": [
        "481", "482", "483", "484", "485", "486", "487", "488",
        "491", "492", "493", "541614",
    ],
    "software": [
        "5112", "5182", "5415", "5416",
    ],
    "food_beverage": [
        "311", "312", "4451", "4452", "4453", "42441", "42442", "42443",
        "42449",
    ],
    "packaging": [
        "3221", "3222", "3261", "3279", "32651", "332431", "332439",
    ],
    "construction": [
        "236", "237", "238", "541310", "541330",
    ],
    "pharmaceutical": [
        "3254", "4242", "54171", "62151",
    ],
    "hr": [
        "54161", "56132", "56133", "56131",
    ],
    "clean_energy": [
        "2211", "2212", "33361", "33359", "237130",
    ],
    "fintech": [
        "5221", "5222", "5223", "5224", "5231", "5241", "5259", "52232",
        "52239",
    ],
    "ecommerce": [
        "4541", "45411",
    ],
    "renewable_equipment": [
        "33361", "335911", "335912", "335999", "333611",
    ],
    "ev_battery": [
        "335911", "335912", "33331", "3291", "3255", "3313", "3314",
        "334413",
    ],
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _parse_dict_field(value: Any) -> dict:
    """Parse a field that may be stored as a dict, a JSON string, or a Python-repr string."""
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    s = str(value)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, dict) else {}
    except Exception:
        pass
    return {}


def _text_of(value: Any) -> str:
    """Flatten any value to a searchable lowercase string."""
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(_text_of(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_text_of(v) for v in value.values())
    return str(value).lower()


def _get_country_code(company: dict) -> str | None:
    addr = _parse_dict_field(company.get("address"))
    return addr.get("country_code") or None


def _company_text(company: dict) -> str:
    """Build a single searchable text blob for a company."""
    parts: list[str] = []
    for key in [
        "operational_name", "description", "business_model",
        "core_offerings", "target_markets",
    ]:
        parts.append(_text_of(company.get(key)))
    naics = _parse_dict_field(company.get("primary_naics"))
    parts.append(_text_of(naics.get("label", "")))
    secondary = company.get("secondary_naics") or []
    if isinstance(secondary, list):
        for sn in secondary:
            parsed = _parse_dict_field(sn)
            parts.append(_text_of(parsed.get("label", "")))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Query Analyzer
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    # Hard filters
    country_codes: list[str] = field(default_factory=list)   # ISO-2 codes to match
    region_codes: frozenset[str] | None = None                # match any in set
    min_employees: int | None = None
    max_employees: int | None = None
    min_revenue: float | None = None
    max_revenue: float | None = None
    founded_after: int | None = None
    founded_before: int | None = None
    is_public: bool | None = None
    required_business_models: list[str] = field(default_factory=list)

    # Soft / semantic
    industry_groups: list[str] = field(default_factory=list)  # keys from INDUSTRY_KEYWORDS
    semantic_keywords: list[str] = field(default_factory=list)

    # Whether query is "complex" (needs LLM for best results)
    is_complex: bool = False


_NUM_SUFFIXES = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000,
}


def _parse_number(s: str) -> float | None:
    s = s.strip().lower().replace(",", "")
    m = re.match(r"([\d.]+)\s*(k|m|b|thousand|million|billion)?$", s)
    if not m:
        return None
    num = float(m.group(1))
    suf = m.group(2)
    if suf:
        num *= _NUM_SUFFIXES.get(suf, 1)
    return num


def analyze_query(query: str) -> QueryIntent:
    """Extract structured intent and semantic keywords from a natural-language query."""
    intent = QueryIntent()
    q = query.lower()

    # --- Public / private ---
    if re.search(r"\bpublic\b", q):
        intent.is_public = True
    if re.search(r"\bprivate\b", q):
        intent.is_public = False

    # --- Business model ---
    if re.search(r"\bb2b\b|business.to.business", q):
        intent.required_business_models.append("b2b")
    if re.search(r"\bsaas\b|software.as.a.service", q):
        intent.required_business_models.append("saas")

    # --- Employee count ---
    emp_more = re.search(
        r"(?:more than|over|greater than|at least|above)\s+([\d,]+(?:\.\d+)?(?:\s*[kmb])?)\s*employees",
        q,
    )
    if emp_more:
        val = _parse_number(emp_more.group(1))
        if val is not None:
            intent.min_employees = int(val)

    emp_less = re.search(
        r"(?:fewer than|less than|under|below|at most)\s+([\d,]+(?:\.\d+)?(?:\s*[kmb])?)\s*employees",
        q,
    )
    if emp_less:
        val = _parse_number(emp_less.group(1))
        if val is not None:
            intent.max_employees = int(val)

    # --- Revenue ---
    rev_more = re.search(
        r"revenue\s+(?:over|above|more than|greater than|exceeding|of over|of more than)\s+\$?([\d,.]+(?:\s*[kmb])?)",
        q,
    )
    if not rev_more:
        rev_more = re.search(
            r"(?:over|above|more than|greater than)\s+\$\s*([\d,.]+(?:\s*[kmb])?)",
            q,
        )
    if rev_more:
        val = _parse_number(rev_more.group(1))
        if val is not None:
            intent.min_revenue = val

    rev_less = re.search(
        r"revenue\s+(?:under|below|less than|fewer than)\s+\$?([\d,.]+(?:\s*[kmb])?)",
        q,
    )
    if rev_less:
        val = _parse_number(rev_less.group(1))
        if val is not None:
            intent.max_revenue = val

    # --- Founded year ---
    year_after = re.search(
        r"(?:founded|established|started|created)\s+(?:after|since|post|from)\s+(\d{4})",
        q,
    )
    if year_after:
        intent.founded_after = int(year_after.group(1))

    year_before = re.search(
        r"(?:founded|established|started|created)\s+(?:before|prior to|until)\s+(\d{4})",
        q,
    )
    if year_before:
        intent.founded_before = int(year_before.group(1))

    # --- Geography ---
    # Try multi-word countries first (e.g. "united states", "south korea")
    detected_countries: set[str] = set()
    detected_regions: set[str] = set()

    for phrase, code in sorted(COUNTRY_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            detected_countries.add(code)

    for phrase, codes in REGION_MAP.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            detected_regions.update(codes)

    intent.country_codes = list(detected_countries)
    if detected_regions:
        intent.region_codes = frozenset(detected_regions)

    # --- Industry groups ---
    matched_groups: list[str] = []
    for group, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if _kw_match(kw, q):
                if group not in matched_groups:
                    matched_groups.append(group)
                break

    # Second pass: check if any meaningful query token is a prefix of a single-word
    # keyword. This handles "logistic" → matches "logistics", etc.
    # Multi-word keywords (with spaces) are excluded to avoid false positives like
    # "supply" matching "supply chain" or "electric" matching "electrical contractor".
    q_tokens = [t for t in re.findall(r"[a-z][a-z\-']+", q) if len(t) >= 5]
    for group, keywords in INDUSTRY_KEYWORDS.items():
        if group in matched_groups:
            continue
        for kw in keywords:
            if " " in kw:
                continue  # only single-word keywords for prefix matching
            for token in q_tokens:
                if kw.startswith(token) and kw != token:
                    matched_groups.append(group)
                    break
            if group in matched_groups:
                break

    intent.industry_groups = matched_groups

    # --- Residual semantic keywords (words not captured by rules) ---
    # Strip out stop-words and already-captured terms
    stop_words = {
        "a", "an", "the", "and", "or", "in", "of", "for", "to", "with",
        "is", "are", "be", "been", "has", "have", "that", "this", "which",
        "who", "more", "than", "over", "under", "less", "from", "at", "on",
        "by", "as", "its", "it", "they", "them", "their", "we", "our",
        "can", "could", "would", "should", "will", "do", "does", "did",
        "using", "providing", "competing", "growing", "founded", "after",
        "before", "since", "about", "where", "when", "how", "what", "why",
        "companies", "company", "businesses", "business", "firms", "firm",
        "public", "private", "based", "located",
    }
    tokens = re.findall(r"[a-z][a-z\-']+", q)
    semantic = [
        t for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    intent.semantic_keywords = semantic

    # Heuristic: mark complex if no hard geographic filter and no strict numeric constraints
    hard_filters = bool(
        intent.country_codes
        or intent.region_codes
        or intent.min_employees is not None
        or intent.min_revenue is not None
        or intent.is_public is not None
    )
    semantic_only_groups = {"fintech", "ecommerce", "ev_battery", "packaging"}
    has_complex_group = bool(set(intent.industry_groups) & semantic_only_groups)
    intent.is_complex = has_complex_group or (not hard_filters and not intent.industry_groups)

    return intent


# ---------------------------------------------------------------------------
# Structured Filter
# ---------------------------------------------------------------------------

def passes_hard_filters(company: dict, intent: QueryIntent) -> bool:
    """Return False if company violates any hard constraint from the intent."""

    # Country / region filter
    if intent.country_codes or intent.region_codes:
        cc = _get_country_code(company)
        country_ok = False
        if cc:
            if intent.country_codes and cc in intent.country_codes:
                country_ok = True
            if intent.region_codes and cc in intent.region_codes:
                country_ok = True
        if not country_ok:
            return False

    # Employee count
    emp = company.get("employee_count")
    if emp is not None:
        try:
            emp = float(emp)
            if intent.min_employees is not None and emp < intent.min_employees:
                return False
            if intent.max_employees is not None and emp > intent.max_employees:
                return False
        except (TypeError, ValueError):
            pass

    # Revenue
    rev = company.get("revenue")
    if rev is not None:
        try:
            rev = float(rev)
            if intent.min_revenue is not None and rev < intent.min_revenue:
                return False
            if intent.max_revenue is not None and rev > intent.max_revenue:
                return False
        except (TypeError, ValueError):
            pass

    # Founded year
    yr = company.get("year_founded")
    if yr is not None:
        try:
            yr = int(float(yr))
            if intent.founded_after is not None and yr <= intent.founded_after:
                return False
            if intent.founded_before is not None and yr >= intent.founded_before:
                return False
        except (TypeError, ValueError):
            pass

    # Public status
    if intent.is_public is not None:
        pub = company.get("is_public")
        if pub is not None and bool(pub) != intent.is_public:
            return False

    return True


# ---------------------------------------------------------------------------
# Semantic Scorer
# ---------------------------------------------------------------------------

def _kw_match(kw: str, text: str) -> bool:
    """
    Word-boundary-aware keyword match (case-insensitive, pre-lowercased text expected).

    - Short keywords (≤ 3 chars like "ev", "hr"): require both boundaries to avoid
      false matches inside longer words (e.g. "ev" inside "revenue").
    - Longer keywords (≥ 4 chars): use left-boundary only so that prefix forms match
      (e.g. keyword "logistic" matches "logistics" in text).
    """
    if kw not in text:
        return False
    if len(kw) <= 3:
        # Exact word boundaries for short abbreviations
        return bool(re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text))
    # Left boundary only: allows keyword to be a prefix of a longer word
    return bool(re.search(r"(?<!\w)" + re.escape(kw), text))


def _keyword_density(text: str, keywords: list[str]) -> float:
    """Return score 0–1 based on fraction of keywords that appear in text."""
    if not keywords or not text:
        return 0.0
    hits = sum(1 for kw in keywords if _kw_match(kw, text))
    # Use a log scale so that a single strong hit still scores well
    return math.log1p(hits) / math.log1p(len(keywords))


def _naics_code(company: dict) -> str:
    """Extract the primary NAICS code string for a company."""
    naics = _parse_dict_field(company.get("primary_naics"))
    return str(naics.get("code", "") or "")


def _naics_code_matches(naics_code: str, prefixes: list[str]) -> bool:
    """Return True if naics_code starts with any of the given prefixes."""
    return any(naics_code.startswith(p) for p in prefixes)


def score_company(company: dict, intent: QueryIntent) -> float:
    """
    Return a relevance score ≥ 0. Higher = more relevant.

    Weights:
      - Industry group match  (0–40)
      - Core offerings match  (0–20)
      - Semantic keywords     (0–25)
      - Business model match  (0–10)
      - Partial numeric bonus  (0–5)
    """
    score = 0.0
    text = _company_text(company)
    offerings_text = _text_of(company.get("core_offerings"))

    naics_code = _naics_code(company)

    # --- NAICS code match (highest precision signal) ---
    naics_match = False
    for group in intent.industry_groups:
        prefixes = NAICS_PREFIXES.get(group, [])
        if prefixes and naics_code and _naics_code_matches(naics_code, prefixes):
            score += 35
            naics_match = True
            break

    # --- Industry group keyword match ---
    for group in intent.industry_groups:
        keywords = INDUSTRY_KEYWORDS.get(group, [])
        density = _keyword_density(text, keywords)
        # Reduce weight if NAICS already confirmed match to avoid double-counting
        weight = 25 if naics_match else 40
        score += density * weight

    if not intent.industry_groups and intent.semantic_keywords:
        # No group matched; fall back fully to semantic keywords
        density = _keyword_density(text, intent.semantic_keywords)
        score += density * 40

    # --- Core offerings match ---
    if intent.semantic_keywords:
        score += _keyword_density(offerings_text, intent.semantic_keywords) * 20

    # --- Semantic keywords across full text ---
    if intent.semantic_keywords:
        score += _keyword_density(text, intent.semantic_keywords) * 25

    # --- Business model ---
    if intent.required_business_models:
        bm_text = _text_of(company.get("business_model", []))
        for bm in intent.required_business_models:
            if _kw_match(bm, bm_text):
                score += 5

    # --- Partial numeric bonuses (reward companies near query thresholds) ---
    if intent.min_employees is not None:
        emp = company.get("employee_count")
        if emp and float(emp) >= intent.min_employees:
            score += 3
    if intent.max_employees is not None:
        emp = company.get("employee_count")
        if emp and float(emp) <= intent.max_employees:
            score += 3

    return score


# ---------------------------------------------------------------------------
# LLM Reranker (optional – requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def _build_company_summary(company: dict) -> str:
    """Build a compact summary of a company for LLM context."""
    lines: list[str] = []
    if company.get("operational_name"):
        lines.append(f"Name: {company['operational_name']}")
    if company.get("address"):
        addr = _parse_dict_field(company["address"])
        loc_parts = [addr.get("town"), addr.get("region_name"), addr.get("country_code")]
        loc = ", ".join(p for p in loc_parts if p)
        if loc:
            lines.append(f"Location: {loc}")
    if company.get("description"):
        # Truncate to first 300 chars
        lines.append(f"Description: {company['description'][:300]}")
    if company.get("core_offerings"):
        offerings = company["core_offerings"]
        if isinstance(offerings, list):
            lines.append(f"Offerings: {', '.join(str(o) for o in offerings[:5])}")
    return "\n".join(lines)


def llm_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 20,
    model: str = "gpt-4o-mini",
) -> list[tuple[dict, float]]:
    """
    Use an LLM to score candidates.  Returns a list of (company, score) tuples.
    Falls back to an empty list if the API is unavailable.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        print("[LLM] openai package not installed – skipping LLM reranking", file=sys.stderr)
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] OPENAI_API_KEY not set – skipping LLM reranking", file=sys.stderr)
        return []

    client = openai.OpenAI(api_key=api_key)
    batch = candidates[:top_k]

    numbered_summaries = "\n\n".join(
        f"[{i+1}] {_build_company_summary(c)}" for i, c in enumerate(batch)
    )

    prompt = (
        f"You are an expert company analyst. A user is looking for:\n"
        f'"{query}"\n\n'
        f"For each of the {len(batch)} companies below, assign a relevance score from 0 to 10 "
        f"(10 = perfect match, 0 = no match). Consider industry, location, size, and intent.\n\n"
        f"{numbered_summaries}\n\n"
        f"Respond with ONLY a JSON array of numbers, one per company, in the same order.\n"
        f"Example: [8, 3, 9, 0, 7]"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        scores = json.loads(raw)
        if isinstance(scores, list) and len(scores) == len(batch):
            return [(c, float(s)) for c, s in zip(batch, scores)]
    except Exception as exc:
        print(f"[LLM] API call failed: {exc}", file=sys.stderr)

    return []


# ---------------------------------------------------------------------------
# Main Qualification System
# ---------------------------------------------------------------------------

class QualificationSystem:
    """
    Multi-stage pipeline:
      1. Parse query → QueryIntent
      2. Filter companies by hard constraints
      3. Score remaining companies semantically
      4. (Optional) LLM rerank top-K candidates
      5. Return ranked list with scores
    """

    def __init__(
        self,
        companies: list[dict],
        use_llm: bool = False,
        llm_top_k: int = 30,
        min_score_threshold: float = 0.5,
        max_results: int = 50,
    ):
        self.companies = companies
        self.use_llm = use_llm
        self.llm_top_k = llm_top_k
        self.min_score_threshold = min_score_threshold
        self.max_results = max_results

    def qualify(self, query: str) -> list[dict]:
        """
        Process a query and return a ranked list of matching companies.
        Each result dict contains the company fields plus 'score' and 'rank'.
        """
        intent = analyze_query(query)

        # Stage 1: hard filters
        candidates = [c for c in self.companies if passes_hard_filters(c, intent)]

        # Deduplicate: keep first occurrence per (website, operational_name) pair
        seen: set[tuple[str, str]] = set()
        unique_candidates: list[dict] = []
        for c in candidates:
            key = (
                str(c.get("website") or "").lower().strip(),
                str(c.get("operational_name") or "").lower().strip(),
            )
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        candidates = unique_candidates

        # Stage 2: semantic scoring
        scored: list[tuple[dict, float]] = [
            (c, score_company(c, intent)) for c in candidates
        ]

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Stage 3 (optional): LLM rerank
        if self.use_llm and intent.is_complex:
            llm_results = llm_rerank(query, [c for c, _ in scored], top_k=self.llm_top_k)
            if llm_results:
                # Blend LLM score (normalised to same range) with keyword score
                # LLM score: 0–10; keyword score: 0–90+.  Scale LLM to 0–50 and add.
                llm_map: dict[str, float] = {}
                for c, s in llm_results:
                    key = c.get("website", id(c))
                    llm_map[str(key)] = s

                blended: list[tuple[dict, float]] = []
                for c, kw_score in scored:
                    key = str(c.get("website", id(c)))
                    llm_s = llm_map.get(key)
                    if llm_s is not None:
                        final = kw_score + llm_s * 5  # scale 0–10 → 0–50
                    else:
                        final = kw_score
                    blended.append((c, final))
                blended.sort(key=lambda x: x[1], reverse=True)
                scored = blended

        # Filter by threshold and cap results
        filtered = [(c, s) for c, s in scored if s >= self.min_score_threshold]
        top = filtered[: self.max_results]

        results: list[dict] = []
        for rank, (company, score) in enumerate(top, start=1):
            result = dict(company)
            result["score"] = round(score, 3)
            result["rank"] = rank
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------------

DEMO_QUERIES: list[str] = [
    "Logistic companies in Romania",
    "Public software companies with more than 1,000 employees.",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Construction companies in the United States with revenue over $50 million",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Fast-growing fintech companies competing with traditional banks in Europe.",
    "E-commerce companies using Shopify or similar platforms",
    "Renewable energy equipment manufacturers in Scandinavia",
    "Companies that manufacture or supply critical components for electric vehicle battery production",
]


def load_companies(path: str = "companies.jsonl") -> list[dict]:
    companies: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                companies.append(json.loads(line))
    return companies


def _format_result_row(company: dict, rank: int) -> str:
    name = company.get("operational_name") or company.get("website") or "?"
    addr = _parse_dict_field(company.get("address", {}))
    loc = ", ".join(
        p for p in [addr.get("town"), addr.get("country_code")] if p
    )
    score = company.get("score", 0)
    emp = company.get("employee_count")
    emp_str = f"{int(emp):,}" if emp else "?"
    return f"  {rank:3d}. [{score:5.1f}] {name:<35s}  {loc:<20s}  employees: {emp_str}"


def run_queries(
    queries: list[str],
    companies: list[dict],
    use_llm: bool = False,
    output_json: bool = False,
) -> dict[str, list[dict]]:
    system = QualificationSystem(companies, use_llm=use_llm)
    all_results: dict[str, list[dict]] = {}

    for query in queries:
        intent = analyze_query(query)
        results = system.qualify(query)
        all_results[query] = results

        if not output_json:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"  → {len(results)} qualified companies")
            intent_parts: list[str] = []
            if intent.country_codes:
                intent_parts.append(f"countries={intent.country_codes}")
            if intent.region_codes:
                intent_parts.append("region-filter=ON")
            if intent.min_employees is not None:
                intent_parts.append(f"min_employees={intent.min_employees}")
            if intent.max_employees is not None:
                intent_parts.append(f"max_employees={intent.max_employees}")
            if intent.min_revenue is not None:
                intent_parts.append(f"min_revenue={intent.min_revenue:,.0f}")
            if intent.is_public is not None:
                intent_parts.append(f"is_public={intent.is_public}")
            if intent.founded_after is not None:
                intent_parts.append(f"founded_after={intent.founded_after}")
            if intent.founded_before is not None:
                intent_parts.append(f"founded_before={intent.founded_before}")
            if intent.industry_groups:
                intent_parts.append(f"industries={intent.industry_groups}")
            if intent_parts:
                print(f"  Intent: {', '.join(intent_parts)}")
            print()
            for company in results[:15]:
                print(_format_result_row(company, company["rank"]))

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intent Qualification System – rank companies against user queries."
    )
    parser.add_argument(
        "--data",
        default="companies.jsonl",
        help="Path to the JSONL company dataset (default: companies.jsonl)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query to run. If omitted, runs all 12 demo queries.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM reranking for complex queries (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    companies = load_companies(args.data)
    queries = [args.query] if args.query else DEMO_QUERIES

    results = run_queries(
        queries,
        companies,
        use_llm=args.use_llm,
        output_json=args.output_json,
    )

    if args.output_json:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
