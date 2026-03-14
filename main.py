#!/usr/bin/env python3
"""
Entry point – runs the Intent Qualification System on all 12 benchmark queries.

For more options (custom query, JSON output, LLM re-ranking, etc.) use solution.py
directly:

    python solution.py --help
"""

from solution import BENCHMARK_QUERIES, load_companies, build_tfidf_index, qualify, print_results


def main() -> None:
    companies = load_companies("companies.jsonl")
    vectorizer = build_tfidf_index(companies)

    for query in BENCHMARK_QUERIES:
        results = qualify(query, companies, vectorizer)
        print_results(query, results)


if __name__ == "__main__":
    main()
