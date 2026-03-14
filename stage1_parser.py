"""
Stage 1: Intent Deconstruction (The Parser)
Model: DeepSeek V3 via featherless.ai
Input:  Raw natural language query Q
Output: Structured JSON with filters, semantic keywords, and role label
"""

import json
import os
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
    client = OpenAI(api_key=api_key or FEATHERLESS_API_KEY, base_url=FEATHERLESS_BASE_URL)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        temperature=0.0,   # deterministic — we need valid JSON every time
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if the model ignores the instruction
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stage 1 returned invalid JSON:\n{raw}") from exc

    return parsed


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
