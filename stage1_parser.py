"""
Stage 1: Intent Deconstruction (The Parser)
Model: DeepSeek V3 via featherless.ai
Input:  Raw natural language query Q
Output: Structured JSON with filters, semantic keywords, and role label
"""

import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FEATHERLESS_API_KEY = os.environ.get("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3-0324"

# JSON schema that the LLM must populate
PARSER_OUTPUT_SCHEMA = {
    "structured_filters": {
        "country_codes": "list[str] | null  # ISO-2 codes, e.g. ['de','at']",
        "min_employees": "int | null",
        "max_employees": "int | null",
        "min_revenue_usd": "float | null",
        "max_revenue_usd": "float | null",
        "is_public": "bool | null",
        "naics_codes": "list[str] | null  # 6-digit NAICS codes when inferable",
        "business_models": "list[str] | null  # e.g. ['Manufacturing','Wholesale']",
        "target_markets": "list[str] | null  # e.g. ['Automotive','Pharma']",
    },
    "semantic_keywords": "list[str]  # expanded synonyms & related terms for vector search",
    "role_label": "str  # one of: Supplier, Manufacturer, Distributor, Competitor, Customer, Investor, Acquisition Target, Partner, Service Provider, Unknown",
    "reasoning": "str  # one sentence explaining the role classification",
}

SYSTEM_PROMPT = f"""You are a B2B company-search intent parser.

Given a natural language query, extract ALL of the following into a single valid JSON object.
Return ONLY the JSON — no markdown fences, no prose, no extra keys.

JSON Schema (replace descriptions with actual values):
{json.dumps(PARSER_OUTPUT_SCHEMA, indent=2)}

Rules:
1. structured_filters: set a field to null if the query gives no signal for it.
2. semantic_keywords: include the query's core nouns/verbs PLUS 5-10 synonyms or related industry terms that would improve embedding search recall.
3. role_label: infer the RELATIONSHIP the searcher wants with the found companies. 
   - "I need a packaging supplier" → Supplier
   - "competitors in the CRM space" → Competitor
   - "companies to acquire" → Acquisition Target
   - "who buys our steel" → Customer
4. If a field is ambiguous, choose the most probable value and note it in reasoning.
"""


ROLE_LABELS = {
    "Supplier",
    "Manufacturer",
    "Distributor",
    "Competitor",
    "Customer",
    "Investor",
    "Acquisition Target",
    "Partner",
    "Service Provider",
    "Unknown",
}


def _default_parsed(query: str) -> dict:
    q = query.lower()

    country_map = {
        "german": "de",
        "germany": "de",
        "france": "fr",
        "french": "fr",
        "italy": "it",
        "italian": "it",
        "spain": "es",
        "spanish": "es",
        "romania": "ro",
        "romanian": "ro",
        "usa": "us",
        "united states": "us",
        "uk": "gb",
        "united kingdom": "gb",
    }

    detected_countries = [
        code for token, code in country_map.items() if token in q
    ]
    country_codes = sorted(set(detected_countries)) or None

    if any(k in q for k in ["competitor", "competition", "rival"]):
        role_label = "Competitor"
    elif any(k in q for k in ["acquire", "acquisition", "buy company", "m&a"]):
        role_label = "Acquisition Target"
    elif any(k in q for k in ["investor", "investment", "fund"]):
        role_label = "Investor"
    elif any(k in q for k in ["customer", "buyer", "who buys"]):
        role_label = "Customer"
    elif any(k in q for k in ["partner", "alliance"]):
        role_label = "Partner"
    elif any(k in q for k in ["service provider", "agency", "consulting"]):
        role_label = "Service Provider"
    elif "distributor" in q:
        role_label = "Distributor"
    elif "manufacturer" in q:
        role_label = "Manufacturer"
    elif any(k in q for k in ["supplier", "vendor", "provider"]):
        role_label = "Supplier"
    else:
        role_label = "Unknown"

    words = re.findall(r"[a-z0-9]+", q)
    keep = [w for w in words if len(w) >= 4]
    semantic_keywords = list(dict.fromkeys(keep))[:10]

    return {
        "structured_filters": {
            "country_codes": country_codes,
            "min_employees": None,
            "max_employees": None,
            "min_revenue_usd": None,
            "max_revenue_usd": None,
            "is_public": None,
            "naics_codes": None,
            "business_models": None,
            "target_markets": None,
        },
        "semantic_keywords": semantic_keywords,
        "role_label": role_label,
        "reasoning": "Fallback parser used because the model output was not valid JSON.",
    }


def _extract_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(raw)):
        char = raw[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start:idx + 1]
    return None


def _normalise_parsed(parsed: dict, query: str) -> dict:
    base = _default_parsed(query)

    structured = parsed.get("structured_filters") if isinstance(parsed, dict) else None
    if isinstance(structured, dict):
        for key in base["structured_filters"]:
            if key in structured:
                base["structured_filters"][key] = structured[key]

    keywords = parsed.get("semantic_keywords") if isinstance(parsed, dict) else None
    if isinstance(keywords, list):
        base["semantic_keywords"] = keywords

    role_label = parsed.get("role_label") if isinstance(parsed, dict) else None
    if isinstance(role_label, str) and role_label in ROLE_LABELS:
        base["role_label"] = role_label

    reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
    if isinstance(reasoning, str) and reasoning.strip():
        base["reasoning"] = reasoning.strip()

    return base


def parse_query(query: str, api_key: str = FEATHERLESS_API_KEY) -> dict:
    """
    Call DeepSeek V3 on featherless.ai to deconstruct a natural language query
    into structured filters, semantic keywords, and a role label.

    Args:
        query: Raw natural language search query from the user.
        api_key: featherless.ai API key (falls back to FEATHERLESS_API_KEY env var).

    Returns:
        Parsed dict with keys: structured_filters, semantic_keywords, role_label, reasoning.

    Raises:
        ValueError: If the model returns invalid JSON.
        openai.APIError: On API-level errors.
    """
    resolved_api_key = api_key or FEATHERLESS_API_KEY
    if not resolved_api_key:
        print("WARNING: FEATHERLESS_API_KEY not set. Using fallback parser.")
        return _default_parsed(query)

    client = OpenAI(api_key=resolved_api_key, base_url=FEATHERLESS_BASE_URL)

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"},
    ]

    for attempt in range(2):
        messages = base_messages
        if attempt == 1:
            messages = base_messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid. "
                        "Return exactly one valid JSON object matching the schema, no prose."
                    ),
                }
            ]

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = (response.choices[0].message.content or "").strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        for candidate in [raw, _extract_json_object(raw)]:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                return _normalise_parsed(parsed, query)
            except json.JSONDecodeError:
                continue

    print("WARNING: Stage 1 returned invalid JSON twice. Using fallback parser.")
    return _default_parsed(query)


def format_filters_for_display(parsed: dict) -> str:
    """Return a human-readable summary of the parsed filters."""
    sf = parsed.get("structured_filters", {})
    lines = ["=== Stage 1 Parser Output ==="]
    lines.append(f"Role Label      : {parsed.get('role_label', 'Unknown')}")
    lines.append(f"Reasoning       : {parsed.get('reasoning', '')}")
    lines.append(f"Semantic KWs    : {', '.join(parsed.get('semantic_keywords', []))}")
    lines.append("Structured Filters:")
    for k, v in sf.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Find me German packaging suppliers with more than 500 employees "
        "that sell to the food & beverage industry"
    )

    print(f"Query: {query}\n")
    result = parse_query(query)
    print(format_filters_for_display(result))
    print("\nRaw JSON:")
    print(json.dumps(result, indent=2))
