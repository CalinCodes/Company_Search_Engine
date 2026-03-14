"""
Stage 1 Filter: applies the structured_filters from the Stage 1 parser
against transformed_data.json and writes the surviving candidates to
processed1.json.

Each output record is annotated with a `_filter_match` dict that shows
which filters were applied and whether each matched, so the result is
fully traceable.
"""

import json
from typing import Any


def _to_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _normalise_str(s: str) -> str:
    return s.lower().strip() if isinstance(s, str) else ""


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _escape_sql_literal(text: str) -> str:
    return text.replace("'", "''")


def _matches_country_codes(company: dict, value: Any) -> bool:
    countries = {_normalise_str(c) for c in _to_list(value)}
    co_country = _normalise_str(company.get("address_country_code"))
    return co_country in countries


def _matches_min_employees(company: dict, value: Any) -> bool:
    emp = _to_float_or_none(company.get("employee_count"))
    min_emp = _to_float_or_none(value)
    return emp is not None and min_emp is not None and emp >= min_emp


def _matches_max_employees(company: dict, value: Any) -> bool:
    emp = _to_float_or_none(company.get("employee_count"))
    max_emp = _to_float_or_none(value)
    return emp is not None and max_emp is not None and emp <= max_emp


def _matches_min_revenue(company: dict, value: Any) -> bool:
    rev = _to_float_or_none(company.get("revenue"))
    min_rev = _to_float_or_none(value)
    return rev is not None and min_rev is not None and rev >= min_rev


def _matches_max_revenue(company: dict, value: Any) -> bool:
    rev = _to_float_or_none(company.get("revenue"))
    max_rev = _to_float_or_none(value)
    return rev is not None and max_rev is not None and rev <= max_rev


def _matches_is_public(company: dict, value: Any) -> bool:
    return company.get("is_public") == bool(value)


def _matches_naics_codes(company: dict, value: Any) -> bool:
    requested_codes = [str(v).strip() for v in _to_list(value) if str(v).strip()]
    co_naics = str(company.get("primary_naics_code") or "")
    return any(co_naics.startswith(code) for code in requested_codes)


def _matches_business_models(company: dict, value: Any) -> bool:
    requested = {_normalise_str(v) for v in _to_list(value)}
    co_models = {_normalise_str(v) for v in _to_list(company.get("business_model"))}
    return bool(requested & co_models)


def _matches_target_markets(company: dict, value: Any) -> bool:
    requested = [_normalise_str(v) for v in _to_list(value)]
    co_markets = " ".join(_normalise_str(v) for v in _to_list(company.get("target_markets")))
    return any(token in co_markets for token in requested)


FILTER_HANDLERS = {
    "country_codes": _matches_country_codes,
    "min_employees": _matches_min_employees,
    "max_employees": _matches_max_employees,
    "min_revenue_usd": _matches_min_revenue,
    "max_revenue_usd": _matches_max_revenue,
    "is_public": _matches_is_public,
    "naics_codes": _matches_naics_codes,
    "business_models": _matches_business_models,
    "target_markets": _matches_target_markets,
}


def build_sql_query(filters: dict, table_name: str = "companies") -> str:
    """
    Build a SQL-like WHERE query from non-null structured filters.
    This is used for traceability/debugging and mirrors apply_filters logic.
    """
    where_clauses: list[str] = []

    for key, value in filters.items():
        if not _is_set(value):
            continue

        if key == "country_codes":
            countries = [str(c).strip().lower() for c in _to_list(value) if str(c).strip()]
            if countries:
                joined = ", ".join(f"'{_escape_sql_literal(c)}'" for c in countries)
                where_clauses.append(f"LOWER(address_country_code) IN ({joined})")
        elif key == "min_employees":
            where_clauses.append(f"employee_count >= {float(value)}")
        elif key == "max_employees":
            where_clauses.append(f"employee_count <= {float(value)}")
        elif key == "min_revenue_usd":
            where_clauses.append(f"revenue >= {float(value)}")
        elif key == "max_revenue_usd":
            where_clauses.append(f"revenue <= {float(value)}")
        elif key == "is_public":
            where_clauses.append(f"is_public = {'TRUE' if value else 'FALSE'}")
        elif key == "naics_codes":
            codes = [str(c).strip() for c in _to_list(value) if str(c).strip()]
            if codes:
                joined = " OR ".join(
                    f"CAST(primary_naics_code AS TEXT) LIKE '{_escape_sql_literal(code)}%'"
                    for code in codes
                )
                where_clauses.append(f"({joined})")
        elif key == "business_models":
            models = [str(m).strip().lower() for m in _to_list(value) if str(m).strip()]
            if models:
                joined = " OR ".join(
                    f"LOWER(CAST(business_model AS TEXT)) LIKE '%{_escape_sql_literal(m)}%'"
                    for m in models
                )
                where_clauses.append(f"({joined})")
        elif key == "target_markets":
            markets = [str(m).strip().lower() for m in _to_list(value) if str(m).strip()]
            if markets:
                joined = " OR ".join(
                    f"LOWER(CAST(target_markets AS TEXT)) LIKE '%{_escape_sql_literal(m)}%'"
                    for m in markets
                )
                where_clauses.append(f"({joined})")

    base_query = f"SELECT * FROM {table_name}"
    return f"{base_query} WHERE {' AND '.join(where_clauses)}" if where_clauses else base_query


def apply_filters(companies: list[dict], filters: dict) -> list[dict]:
    """
    Hard-filter a list of company dicts using the structured_filters produced
    by Stage 1.  A company is kept only if ALL non-null filters pass.

    Returns a list of dicts identical to the input records but with an extra
    `_filter_match` key that records which filters were evaluated.
    """
    results = []

    active_filters = {
        key: value
        for key, value in (filters or {}).items()
        if _is_set(value)
    }

    for co in companies:
        match_log: dict[str, bool | str] = {}
        keep = True

        for key, expected_value in active_filters.items():
            handler = FILTER_HANDLERS.get(key)
            if handler is None:
                # Unknown filter keys are ignored in filtering but logged for traceability.
                match_log[key] = "ignored_unknown_filter"
                continue

            passed = handler(co, expected_value)
            match_log[key] = passed
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
    sql_query = build_sql_query(filters)

    output = {
        "query_parsed": query_parsed,
        "stage1_sql_query": sql_query,
        "total_input": len(companies),
        "total_output": len(filtered),
        "companies": filtered,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Stage 1 filter: {len(companies)} → {len(filtered)} companies → {output_path}")
    print(f"Stage 1 SQL-like query: {sql_query}")
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
