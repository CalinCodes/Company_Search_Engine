"""
Stage 3: LLM Final Filter (Qwen/Qwen2.5-7B-Instruct)

Takes the ranked candidates from Stage 2 (processed2.json) and uses
Qwen2.5-7B-Instruct via featherless.ai to perform a final relevance pass
against the original natural-language query.

Companies are evaluated in batches; only those the model judges as relevant
are kept in the output (processed3.json).
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FEATHERLESS_API_KEY = os.environ.get("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"

BATCH_SIZE = 10   # companies per API call


def _company_summary(co: dict, idx: int) -> str:
    """Build a compact text profile for a single company."""
    lines = [f"[{idx}] {co.get('operational_name', 'N/A')}"]

    desc = co.get("description", "")
    if desc:
        lines.append(f"  Description: {desc[:300]}")

    naics = co.get("primary_naics_label")
    if naics:
        lines.append(f"  Industry: {naics}")

    offerings = co.get("core_offerings") or []
    if offerings:
        lines.append(f"  Offerings: {', '.join(str(o) for o in offerings[:5])}")

    bm = co.get("business_model") or []
    if bm:
        lines.append(f"  Business Model: {', '.join(str(b) for b in bm)}")

    tm = co.get("target_markets") or []
    if tm:
        lines.append(f"  Target Markets: {', '.join(str(t) for t in tm)}")

    meta = []
    country = co.get("address_country_code", "")
    if country:
        meta.append(f"Country: {country.upper()}")
    emp = co.get("employee_count")
    if emp:
        meta.append(f"Employees: {int(emp)}")
    if meta:
        lines.append(f"  {', '.join(meta)}")

    return "\n".join(lines)


def _evaluate_batch(
    client: OpenAI,
    original_query: str,
    companies: list[dict],
    start_idx: int,
    query_parsed: dict | None = None,
) -> list[dict]:
    """
    Ask Qwen2.5-7B-Instruct to judge relevance for a batch of companies.
    Returns a list of {id, relevant, reason} dicts.
    """
    company_texts = "\n\n".join(
        _company_summary(co, start_idx + i)
        for i, co in enumerate(companies)
    )

    indices = list(range(start_idx, start_idx + len(companies)))

    system_prompt = (
        "You are a B2B company-search relevance evaluator. "
        "Given a user's search query and a list of companies, decide which companies "
        "are reasonably relevant to the query intent.\n\n"
        "Evaluation priorities (in order):\n"
        "1. **Industry match is the most important signal.** Judge whether the company operates "
        "in the industry/sector the query describes. Interpret industry terms broadly: e.g. "
        "'software company' includes product companies, service providers, consultancies, SaaS, "
        "and outsourcing firms that work in software. Do NOT split hairs between sub-categories "
        "within the same industry.\n"
        "2. **Business model alignment.** If the query specifies a business model "
        "(e.g. manufacturer, supplier, distributor), check that the company's business model "
        "is compatible. A 'Service Provider' in the right industry still counts.\n"
        "3. Hard filters like geography, company size, or public/private status — only filter "
        "on these if they are explicitly stated in the query AND the company clearly violates them.\n\n"
        "Be inclusive: when in doubt, keep the company. Only filter out companies whose core "
        "business is in a completely different industry from what the query asks for.\n"
        "For each company return whether it is relevant (true/false) and a one-sentence reason.\n\n"
        "Return ONLY a valid JSON array — no prose, no markdown fences. Example:\n"
        '[{"id":0,"relevant":true,"reason":"..."},{"id":1,"relevant":false,"reason":"..."}]'
    )

    # Build extra context from parsed query (business models, role)
    extra_context = ""
    if query_parsed:
        filters = query_parsed.get("structured_filters") or {}
        bm = filters.get("business_models")
        if bm:
            extra_context += f"Desired business models: {', '.join(bm)}\n"
        role = query_parsed.get("role_label")
        if role and role != "Unknown":
            extra_context += f"Desired company role: {role}\n"

    user_content = (
        f'Original search query: "{original_query}"\n'
        f"{extra_context}\n"
        f"Companies to evaluate:\n\n{company_texts}\n\n"
        f"Return a JSON array with one entry per company, IDs {indices[0]} through {indices[-1]}."
    )

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    raw = (response.choices[0].message.content or "").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Extract the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]

    try:
        judgments = json.loads(raw)
        if isinstance(judgments, list):
            return judgments
    except json.JSONDecodeError:
        pass

    # Fallback: keep all companies in this batch if parsing fails
    print(
        f"  WARNING: Could not parse model response for batch starting at {start_idx}. "
        "Keeping all companies in this batch."
    )
    return [
        {"id": start_idx + i, "relevant": True, "reason": "parse-fallback"}
        for i in range(len(companies))
    ]


def run(
    query_parsed: dict,
    input_path: str = "processed2.json",
    output_path: str = "processed3.json",
    api_key: str = "",
) -> list[dict]:
    """
    Load Stage 2 output, run Qwen2.5-7B-Instruct relevance filter, write processed3.json.

    Args:
        query_parsed:  The dict returned by stage1_parser.parse_query().
        input_path:    Path to Stage 2 output JSON.
        output_path:   Where to write Stage 3 output JSON.
        api_key:       featherless.ai API key (falls back to FEATHERLESS_API_KEY env var).

    Returns:
        List of company dicts that passed the final filter.
    """
    resolved_key = api_key or FEATHERLESS_API_KEY
    if not resolved_key:
        raise ValueError("FEATHERLESS_API_KEY is required for Stage 3.")

    with open(input_path) as f:
        stage2 = json.load(f)

    companies = stage2.get("companies", [])
    original_query = query_parsed.get("original_query", "")

    if not companies:
        print("Stage 3: no candidates from Stage 2 — nothing to filter.")
        _write(output_path, query_parsed, [], 0)
        return []

    print(f"Stage 3 (Qwen2.5-7B-Instruct filter): evaluating {len(companies)} candidates …")
    print(f"  Query: {original_query[:120]}")

    client = OpenAI(api_key=resolved_key, base_url=FEATHERLESS_BASE_URL)

    # Collect relevance judgments (idx → judgment dict)
    relevance: dict[int, dict] = {}

    for batch_start in range(0, len(companies), BATCH_SIZE):
        batch = companies[batch_start : batch_start + BATCH_SIZE]
        print(f"  Batch {batch_start}–{batch_start + len(batch) - 1} …")
        judgments = _evaluate_batch(client, original_query, batch, batch_start, query_parsed)
        for j in judgments:
            idx = j.get("id")
            if idx is not None:
                relevance[idx] = j

    # Filter companies
    passed = []
    for i, co in enumerate(companies):
        judgment = relevance.get(i, {})
        is_relevant = judgment.get("relevant", True)  # keep if judgment missing
        reason = judgment.get("reason", "")
        if is_relevant:
            passed.append({**co, "_stage3": {"relevant": True, "reason": reason}})
        else:
            print(f"  Filtered out: {co.get('operational_name', 'N/A')} — {reason}")

    _write(output_path, query_parsed, passed, len(companies))

    print(
        f"Stage 3: {len(companies)} → {len(passed)} companies "
        f"({len(companies) - len(passed)} filtered out) → {output_path}"
    )
    return passed


def _write(
    output_path: str,
    query_parsed: dict,
    companies: list[dict],
    total_input: int,
) -> None:
    output = {
        "query_parsed": query_parsed,
        "model": QWEN_MODEL,
        "total_input": total_input,
        "total_output": len(companies),
        "companies": companies,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "German packaging suppliers for food and beverage"
    )

    dummy_parsed = {
        "original_query": query,
        "structured_filters": {},
        "semantic_keywords": ["packaging", "supplier", "food", "beverage", "Germany"],
        "role_label": "Supplier",
        "reasoning": "Self-test",
    }

    results = run(dummy_parsed)
    for rank, co in enumerate(results, 1):
        s3 = co.get("_stage3", {})
        print(
            f"{rank:2}. {co.get('operational_name', 'N/A'):40s}  "
            f"[{co.get('address_country_code', '').upper()}]  "
            f"reason: {s3.get('reason', '')[:80]}"
        )
