"""
Entry point — demonstrates the four-stage pipeline.
Currently implements Stage 1: Intent Deconstruction + hard filtering.
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from stage1_parser import parse_query, format_filters_for_display
from stage1_filter import run as stage1_filter
from stage2_retrieval import run as stage2_retrieval


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Find me German packaging suppliers "
    )

    api_key = os.environ.get("FEATHERLESS_API_KEY", "")
    if not api_key:
        print("ERROR: FEATHERLESS_API_KEY not set. Add it to .env or export it.\n")
        sys.exit(1)

    print(f"Query: {query}\n")

    # ── Stage 1a: Intent Deconstruction ───────────────────────────────────────
    parsed = parse_query(query, api_key=api_key)
    parsed["original_query"] = query          # passed through to Stage 2+
    print(format_filters_for_display(parsed))

    # ── Stage 1b: Hard Filter → processed1.json ───────────────────────────────
    filtered = stage1_filter(parsed)

    # ── Stage 2: Hybrid Retrieval → processed2.json ───────────────────────────
    ranked = stage2_retrieval(parsed)

    # Stage 3 (Enrichment), Stage 4 (Discriminator)
    # will be wired here in subsequent implementation phases.


if __name__ == "__main__":
    main()

