"""CLI entry point for running the backend pipeline without the HTML UI."""

import argparse
import copy
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from stage1_parser import parse_query, format_filters_for_display
from stage1_filter import run as stage1_filter
import stage2_retrieval


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "final_processed_data.json"
DEFAULT_STAGE1_OUTPUT = BASE_DIR / "processed1.json"
DEFAULT_STAGE2_OUTPUT = BASE_DIR / "processed2.json"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the backend pipeline directly from the terminal.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Natural language search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many ranked companies to print.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the input company dataset JSON file.",
    )
    parser.add_argument(
        "--stage1-output",
        default=str(DEFAULT_STAGE1_OUTPUT),
        help="Where to write Stage 1 filtered results.",
    )
    parser.add_argument(
        "--stage2-output",
        default=str(DEFAULT_STAGE2_OUTPUT),
        help="Where to write Stage 2 ranked results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print ranked results as JSON.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding ranking and use BM25 only.",
    )
    return parser.parse_args()


def _display_name(company: dict) -> str:
    return company.get("operational_name") or "N/A"


def print_ranked_results(ranked: list[dict]) -> None:
    print("\n=== Top Results ===")
    if not ranked:
        print("No companies matched the query.")
        return

    for index, company in enumerate(ranked, start=1):
        retrieval = company.get("_retrieval", {})
        score = retrieval.get("stage2_score", 0.0)
        country = (company.get("address_country_code") or "").upper() or "N/A"
        print(f"{index:2}. [{score:.3f}] {_display_name(company):50s} {country}")


def _relaxed_query(parsed: dict) -> dict:
    relaxed = copy.deepcopy(parsed)
    filters = relaxed.setdefault("structured_filters", {})

    # Keep only geographic/public constraints; relax semantic hard filters.
    for key in [
        "business_models",
        "target_markets",
        "naics_codes",
        "min_employees",
        "max_employees",
        "min_revenue_usd",
        "max_revenue_usd",
    ]:
        filters[key] = None

    relaxed["reasoning"] = (
        f"{relaxed.get('reasoning', '').strip()} "
        "(Auto-relaxed filters after zero Stage 1 matches.)"
    ).strip()
    return relaxed


def main():
    args = build_args()
    query = " ".join(args.query).strip() if args.query else (
        "Find me German packaging suppliers"
    )
    input_path = Path(args.input)
    stage1_output = Path(args.stage1_output)
    stage2_output = Path(args.stage2_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if args.no_embeddings:
        stage2_retrieval.USE_EMBEDDINGS = False

    print(f"Query: {query}\n")

    # ── Stage 1a: Intent Deconstruction ───────────────────────────────────────
    parsed = parse_query(query)
    parsed["original_query"] = query
    print(format_filters_for_display(parsed))

    # ── Stage 1b: Hard Filter → processed1.json ───────────────────────────────
    filtered = stage1_filter(
        parsed,
        input_path=str(input_path),
        output_path=str(stage1_output),
    )

    if not filtered:
        print("Stage 1 returned 0 matches. Retrying with relaxed filters...\n")
        parsed = _relaxed_query(parsed)
        print(format_filters_for_display(parsed))
        filtered = stage1_filter(
            parsed,
            input_path=str(input_path),
            output_path=str(stage1_output),
        )

    # ── Stage 2: Hybrid Retrieval → processed2.json ───────────────────────────
    ranked = stage2_retrieval.run(
        parsed,
        input_path=str(stage1_output),
        output_path=str(stage2_output),
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(ranked, indent=2, default=str))
    else:
        print_ranked_results(ranked)

    print(f"\nStage 1 output: {stage1_output}")
    print(f"Stage 2 output: {stage2_output}")

    # Stage 3 (Enrichment), Stage 4 (Discriminator)
    # will be wired here in subsequent implementation phases.


if __name__ == "__main__":
    main()

