# Intent Qualification System — Writeup

## Approach

### Architecture

The solution is a **multi-stage pipeline** that avoids the extremes of sending every company to an LLM or relying purely on embedding similarity:

```
User Query
    │
    ▼
┌──────────────────┐
│  1. QueryAnalyzer │  → QueryIntent (structured constraints + industry groups + keywords)
└──────────────────┘
         │
         ▼
┌───────────────────────┐
│  2. StructuredFilter   │  → eliminates hard constraint violations (country, revenue, size)
└───────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  3. SemanticScorer                            │
│     ├─ NAICS code match (+35 if industry hit) │  high-precision signal
│     ├─ Industry keyword density in company    │  domain-specific terms
│     ├─ Core offerings keyword match           │  explicit service match
│     └─ Semantic keyword match across text     │  general relevance
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  4. LLM Reranker (optional, --use-llm flag)  │  only top-K, batched
└──────────────────────────────────────────────┘
         │
         ▼
    Ranked results (top 50)
```

### Stage 1 — QueryAnalyzer

Parses the query using a combination of:

- **Regex patterns** for structured constraints:
  - Country / region extraction (e.g., `"in Germany"` → `country_code='de'`)
  - Numeric thresholds (`"more than 1,000 employees"`, `"revenue over $50 million"`)
  - Temporal constraints (`"founded after 2018"`)
  - Public/private flags (`"public companies"`)
  - Business model keywords (`"B2B"`, `"SaaS"`)
- **Industry group detection** via two-pass keyword matching:
  1. *First pass*: direct keyword scan of the lowercased query against 12 industry keyword lists
  2. *Second pass*: query token prefix matching against single-word keywords to catch morphological variants
     (e.g., `"logistic"` in query matches `"logistics"` keyword)
- **Region mapping**: maps named regions to ISO country code sets (`"Europe"`, `"Scandinavia"`)
- **Complexity detection**: marks queries as complex when they contain interpretation-heavy patterns
  (supply chains, competition framing, ecosystem roles), enabling optional LLM reranking

### Stage 2 — StructuredFilter (Hard Filters)

Companies failing any hard constraint are eliminated before scoring:

| Constraint | Example |
|---|---|
| Country code | `"in Romania"` → must have `country_code='ro'` |
| Region | `"in Europe"` → `country_code` in European ISO codes |
| Min employees | `"more than 1,000 employees"` |
| Max employees | `"fewer than 200 employees"` |
| Min revenue | `"revenue over $50 million"` |
| Founded after | `"founded after 2018"` |
| Is public | `"public companies"` → `is_public=True` |

If a field is missing from a company record, the constraint is relaxed for that company (we cannot
exclude what we cannot verify).

### Stage 3 — SemanticScorer

Remaining candidates are scored on a 0–100+ scale using weighted signals:

| Signal | Weight | Notes |
|---|---|---|
| **NAICS code match** | +35 (flat bonus) | High-precision: does the primary NAICS code fall in the right industry category? |
| **Industry keyword density** | ×40 or ×25 (if NAICS matched) | How many domain keywords appear in the full company text? |
| **Core offerings match** | ×20 | Explicit mention in the offerings list |
| **Semantic keyword coverage** | ×25 | Non-stopword query tokens found in company text |
| **Business model match** | +5 per hit | `"B2B"`, `"SaaS"` etc. |
| **Numeric threshold bonus** | +3 | Reward companies that satisfy size constraints |

Key design choices:
- **NAICS code prefix matching**: industry-standard 6-digit NAICS codes are mapped to industry groups
  (e.g., `484xxx` = Trucking → logistics). A NAICS hit gives a flat +35 bonus and reduces keyword
  weight to avoid double-counting.
- **Word-boundary keyword matching**: using `(?<!\w)keyword` (left-boundary) for longer keywords lets
  `"logistic"` match companies mentioning `"logistics"` without spuriously matching short abbreviations
  like `"ev"` inside `"revenue"`.
- **Deduplication**: companies with the same `(website, operational_name)` pair are deduplicated before
  scoring.

### Stage 4 — LLM Reranker (Optional)

For complex queries, the system can call OpenAI (`gpt-4o-mini`) to rerank the top-K candidates
(default K=20). The batch approach keeps costs manageable: one API call per query rather than one per
company. Results are combined with the semantic score (weighted average).

Activated with `--use-llm` flag; requires `OPENAI_API_KEY` environment variable.

---

## Tradeoffs

### What I optimised for

| Priority | Rationale |
|---|---|
| **Accuracy > completeness** | Better to return 8 high-confidence logistics companies than 50 mixed results |
| **Speed (no external calls by default)** | Pure Python, stdlib only, processes ~1000 companies in < 1 s |
| **Cost (zero by default)** | LLM is opt-in; default mode is free |
| **Explainability** | Each result shows a numeric score and the extracted intent, making it easy to debug |

### Intentional tradeoffs

- **Precision over recall**: hard filters eliminate companies with missing fields. A company in Romania
  with no `address.country_code` will be excluded even if it is genuinely Romanian. This avoids noise
  but may miss valid results.
- **Keyword-based rather than embedding-based**: embeddings would rank by *similarity*, which as noted
  in the problem statement confuses cosmetics brands with cosmetics packaging suppliers. Keyword/NAICS
  scoring is more intent-aware.
- **No ML models**: keeps the system dependency-free and deterministic. Accuracy on edge cases is
  lower, but results are consistent and fast.

---

## Error Analysis

### Where the system works well

- **Structured queries**: `"Pharmaceutical companies in Switzerland"` returns Novartis, Bachem,
  Ferring, Lonza near the top — all genuine Swiss pharma companies.
- **Domain-specific packaging queries**: `"packaging for cosmetics brand"` correctly surfaces
  packaging suppliers rather than cosmetics brands (avoiding the embedding similarity failure mode).
- **Startup size filtering**: `"Clean energy startups founded after 2018 with fewer than 200
  employees"` correctly returns small Scandinavian wind/solar startups.

### Where the system struggles

**1. Logistics in Romania** — Companies like OSCAR (petroleum distributor) and METRO România
(wholesale grocery) score moderately because their descriptions contain logistics-adjacent words
(`"distribution"`, `"supply chain"`). They are not logistics companies but use logistics services.
Root cause: keyword-based matching cannot distinguish _"we are a logistics company"_ from _"we use
logistics"_. Mitigation: the NAICS code for true logistics companies (484, 488, 492, 493) separates
them from these false positives.

**2. E-commerce / Shopify query** — The query `"E-commerce companies using Shopify or similar
platforms"` is ambiguous: it could mean (a) brands that sell via Shopify, or (b) Shopify app
developers, or (c) e-commerce platform builders. The system interprets it as e-commerce + software,
which produces a mix of retailers and tech companies. No company in the dataset explicitly states
_"we use Shopify"_ in its description, so the query relies on general e-commerce signals.

**3. "Fast-growing fintech competing with traditional banks"** — The query requires inferring that a
company is "competing with banks" from its product description. The keyword list for fintech covers
payments, neobanks, and digital wallets, but "growth rate" is unobservable from static profile data.
The system returns plausible fintech companies but cannot verify "fast-growing".

---

## Scaling

If the system needed to handle **100,000 companies per query**:

1. **Pre-compute NAICS index**: build a dict `{naics_prefix → [company_ids]}` at load time.
   Hard-filter + NAICS lookup reduces candidates from 100K to ~1K in microseconds.

2. **Inverted keyword index**: map each keyword → set of company IDs that contain it. Scoring
   becomes a set intersection operation (O(k) where k = number of keywords).

3. **Async LLM reranking**: for complex queries, send top-20 candidates to the LLM in a single
   async batch call. The LLM stage doesn't scale with dataset size — only with result count.

4. **Caching**: cache `QueryIntent` objects for repeated or similar queries (LRU or semantic cache).

5. **Vector pre-filtering** (optional): add an ANN (approximate nearest-neighbour) index
   (e.g. FAISS, Qdrant) as a *pre-filter* stage to reduce the candidate pool before the keyword
   scorer runs. This replaces full dataset scanning.

The current approach scans all companies linearly; with the inverted index + NAICS lookup, 100K
companies would still process in under a second.

---

## Failure Modes

### Confident but incorrect results

- **Pharma distributors vs. pharma manufacturers**: the query `"Pharmaceutical companies in
  Switzerland"` does not specify manufacturers. A drug distributor will score identically to a
  manufacturer if both mention `"pharmaceutical"` throughout their text.
- **Subsidiary confusion**: a large multinational's Swiss subsidiary may appear in results even
  if the question targets local Swiss pharma companies.
- **Country code absent**: companies without a `country_code` in their address pass through
  geographic filters silently (relaxed constraint), potentially polluting results for country-specific
  queries.
- **Keyword presence vs. focus**: `"OMV Romania"` (petroleum company) appears in logistics results
  because its description mentions `"logistics"` and `"supply chain"` — even though logistics is not
  its core business. A pure LLM classifier would distinguish this; keyword scoring cannot.

### Production monitoring

| Signal | Alert |
|---|---|
| Mean score of top-10 results drops below threshold | Model/data drift |
| High variance in scores between queries of same type | Keyword list coverage gap |
| Large fraction of results from single country in non-geographic queries | Country-code parsing bug |
| LLM reranker changes top-3 order frequently | Semantic scorer is unreliable for that query type |

---

## Critical Reflection

**Strengths**:
- Deterministic and explainable — scores can be decomposed into contributions from each signal
- Fast (< 1 s for 1000 companies, stdlib only)
- NAICS code matching adds a structured, high-precision signal absent from both naive LLM and
  embedding approaches
- Two-stage filtering (hard → soft) avoids wasting compute on obviously irrelevant companies

**Weaknesses**:
- Keyword coverage is manually curated — new industries require updating the keyword lists
- Cannot infer "fast-growing", "competing with banks", "critical component" from static data
- Missing fields (employees, revenue, year_founded) reduce filter coverage
- Word overlap ≠ business focus: a petroleum distributor that mentions logistics scores similarly
  to a dedicated logistics provider

**What I would prioritise next**:
1. **LLM-based intent extraction** (replace regex parser) — more robust for complex/ambiguous queries
2. **Company embedding index** with *intent-aware* query rewriting (describe what a matching company
   looks like, not just keywords) to avoid the cosmetics/packaging confusion
3. **Feedback loop**: use human labels on a sample of results to calibrate score thresholds
4. **NAICS code coverage expansion**: add more NAICS prefixes per industry group for better recall
