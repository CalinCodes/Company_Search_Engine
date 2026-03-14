"""
Stage 1 Filter: applies the structured_filters from the Stage 1 parser
against transformed_data.json and writes the surviving candidates to
processed1.json.

Each output record is annotated with a `_filter_match` dict that shows
which filters were applied and whether each matched, so the result is
fully traceable.
"""

import json
import re
from typing import Any


def _to_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _normalise_str(s: str) -> str:
    return s.lower().strip() if isinstance(s, str) else ""


def apply_filters(companies: list[dict], filters: dict) -> list[dict]:
    """
    Hard-filter a list of company dicts using the structured_filters produced
    by Stage 1.  A company is kept only if ALL non-null filters pass.

    Returns a list of dicts identical to the input records but with an extra
    `_filter_match` key that records which filters were evaluated.
    """
    results = []

    country_codes = [c.lower() for c in _to_list(filters.get("country_codes"))]
    min_emp = filters.get("min_employees")
    max_emp = filters.get("max_employees")
    min_rev = filters.get("min_revenue_usd")
    max_rev = filters.get("max_revenue_usd")
    is_public = filters.get("is_public")
    naics_codes = _to_list(filters.get("naics_codes"))
    business_models = [_normalise_str(b) for b in _to_list(filters.get("business_models"))]
    target_markets = [_normalise_str(t) for t in _to_list(filters.get("target_markets"))]

    for co in companies:
        match_log: dict[str, bool | str] = {}
        keep = True

        # ── Country ──────────────────────────────────────────────────────────
        if country_codes:
            co_country = _normalise_str(co.get("address_country_code"))
            passed = co_country in country_codes
            match_log["country"] = passed
            if not passed:
                keep = False

        # ── Employee count ────────────────────────────────────────────────────
        emp = co.get("employee_count")
        if min_emp is not None:
            passed = emp is not None and emp >= min_emp
            match_log["min_employees"] = passed
            if not passed:
                keep = False
        if max_emp is not None:
            passed = emp is not None and emp <= max_emp
            match_log["max_employees"] = passed
            if not passed:
                keep = False

        # ── Revenue ───────────────────────────────────────────────────────────
        rev = co.get("revenue")
        if min_rev is not None:
            passed = rev is not None and rev >= min_rev
            match_log["min_revenue_usd"] = passed
            if not passed:
                keep = False
        if max_rev is not None:
            passed = rev is not None and rev <= max_rev
            match_log["max_revenue_usd"] = passed
            if not passed:
                keep = False

        # ── is_public ─────────────────────────────────────────────────────────
        if is_public is not None:
            passed = co.get("is_public") == is_public
            match_log["is_public"] = passed
            if not passed:
                keep = False

        # ── NAICS codes (prefix match so "3241" matches "324110") ─────────────
        if naics_codes:
            co_naics = str(co.get("primary_naics_code") or "")
            passed = any(co_naics.startswith(nc) for nc in naics_codes)
            match_log["naics_codes"] = passed
            if not passed:
                keep = False

        # ── Business models (any overlap) ─────────────────────────────────────
        if business_models:
            co_bm = [_normalise_str(b) for b in _to_list(co.get("business_model"))]
            passed = bool(set(business_models) & set(co_bm))
            match_log["business_models"] = passed
            if not passed:
                keep = False

        # ── Target markets (keyword substring match — more forgiving) ─────────
        if target_markets:
            co_tm = " ".join(_normalise_str(t) for t in _to_list(co.get("target_markets")))
            passed = any(tm in co_tm for tm in target_markets)
            match_log["target_markets"] = passed
            if not passed:
                keep = False

        if keep:
            results.append({**co, "_filter_match": match_log})

    return results


def run(
    query_parsed: dict,
    input_path: str = "final_processed_data.json",
    output_path: str = "processed1.json",
) -> list[dict]:
    """Load companies, apply Stage 1 filters, write processed1.json."""
    with open(input_path) as f:
        companies = json.load(f)

    filters = query_parsed.get("structured_filters", {})
    filtered = apply_filters(companies, filters)

    output = {
        "query_parsed": query_parsed,
        "total_input": len(companies),
        "total_output": len(filtered),
        "companies": filtered,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Stage 1 filter: {len(companies)} → {len(filtered)} companies → {output_path}")
    return filtered


if __name__ == "__main__":
    # Quick self-test with a dummy parsed result (no API call needed)
    dummy = {
        "structured_filters": {
            "country_codes": None,
            "min_employees": 100,
            "max_employees": None,
            "min_revenue_usd": None,
            "max_revenue_usd": None,
            "is_public": None,
            "naics_codes": None,
            "business_models": ["Manufacturing"],
            "target_markets": None,
        },
        "semantic_keywords": ["manufacturer"],
        "role_label": "Supplier",
        "reasoning": "Self-test dummy filters",
    }
    run(dummy)
