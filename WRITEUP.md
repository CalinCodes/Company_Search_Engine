# Intent Qualification System — Writeup

## Approach

### System Architecture

The solution is a **multi-stage qualification pipeline** implemented in `solution.py`. Each stage is designed to be cheap or powerful depending on how far a candidate has progressed through the funnel.

```
User Query
    │
    ▼
┌───────────────────┐
│  Query Analyser   │  Extracts structured constraints (country, size, year, etc.)
│  + Query Cleaner  │  Strips geographic/numeric terms before semantic scoring
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Hard Pre-Filter  │  O(N) pass over all companies — eliminates impossible matches
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stemmed TF-IDF   │  Cosine similarity on rich company text profiles
│  Semantic Scorer  │  Handles morphological variants (logistic ↔ logistics)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Score Boosting   │  Business-model overlap boost (B2B, SaaS, etc.)
│  + Deduplication  │  Remove duplicate domains
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LLM Re-Ranker    │  Optional: OpenAI call on top-20 for complex queries
│  (optional)       │
└───────────────────┘
```

### Component Details

#### 1. Query Analyser (`parse_query`)

Uses regex to extract:
- **Geography**: country names → ISO codes, "Europe" → set of EU country codes, "Scandinavia" → Nordic codes
- **Employee thresholds**: "more than 1,000 employees" → `min_employees=1000`
- **Revenue thresholds**: "$50 million" → `min_revenue=50_000_000`
- **Year founded**: "founded after 2018" → `min_year_founded=2018`
- **Public/private**: "public companies" → `require_public=True`
- **Business models**: "B2B", "SaaS", "B2C" → soft boost in scoring

After extracting structured constraints, geographic/numeric terms are **stripped from the query** before semantic scoring. This prevents `TF-IDF` from matching a French food company to a "France fintech" query simply because it mentions France.

#### 2. Hard Pre-Filter (`passes_filters`)

Eliminates companies that cannot satisfy the query's hard constraints. This is an O(N) scan and typically reduces the candidate pool by 60–90%, dramatically speeding up the expensive scoring step.

#### 3. Stemmed TF-IDF Index (`build_tfidf_index`)

Each company is converted into a rich, flat **text profile** by concatenating:
- NAICS labels (repeated 3× for emphasis)
- Core offerings (repeated 2×)
- Target markets (repeated 2×)
- Business model tokens (repeated 2×)
- Company description
- Name and website
- Geographic tokens (region, town, country code)

The TF-IDF vectorizer uses a **custom stemming tokenizer** (NLTK PorterStemmer) so that morphological variants ("logistic" ↔ "logistics", "pharmaceutical" ↔ "pharmaceuticals") are treated identically in both the index and the query.

A curated set of **extra stop words** (vague terms like "traditional", "growing", "solutions") is zeroed out in the *query vector* (not the corpus) to prevent incidental matches. For example, without this, "fast-growing fintech" would rank a French food company highly because it mentions "crop growing".

#### 4. Score Boosting and Deduplication

After TF-IDF scoring, companies are boosted if their business model overlaps with terms explicitly mentioned in the query (B2B, SaaS, B2C). This is a lightweight +15% boost per matching model.

Companies sharing the same website domain are deduplicated, retaining the highest-scoring entry.

#### 5. LLM Re-Ranking (optional, `llm_qualify`)

When the `--llm` flag is passed and `OPENAI_API_KEY` is set, the top-20 TF-IDF candidates are sent in a single batch request to `gpt-4o-mini`. The model scores each candidate 0–10 and the list is re-ordered accordingly.

This avoids the "LLM per company" antipattern by using TF-IDF as a cheap first-pass filter, only invoking the LLM for the already-promising subset.

---

## Tradeoffs

### What we optimised for

| Priority | Rationale |
|---|---|
| **Accuracy on structured queries** | ~60% of real-world queries have at least one hard constraint (country, size, public). Hard-filtering handles these cheaply and reliably. |
| **Speed** | The TF-IDF index is built once and reused. Scoring 477 companies takes < 100 ms per query on a laptop. |
| **Cost** | No LLM calls by default. LLM is strictly opt-in and limited to top-20 per query. |
| **Robustness to missing data** | All structured filters are `None`-safe; missing fields are simply not filtered on. |

### Deliberate trade-offs

- **TF-IDF over sentence embeddings**: Embeddings (e.g. `sentence-transformers`) would give better semantic understanding, but add a large dependency (~500 MB) and require GPU for fast inference at scale. TF-IDF is still remarkably effective when the corpus is domain-specific.

- **Stemming over lemmatisation**: PorterStemmer is fast and dependency-light but can over-stem (e.g. "pharmaceutical" → "pharmaceut"). Lemmatisation (spaCy) would be more accurate but adds a heavy dependency.

- **Custom stop words over learned filtering**: We zero out vague query terms manually rather than learning which terms cause false positives. This is fragile but transparent and easy to tune.

---

## Error Analysis

### Where the system works well

- **Highly structured queries** ("Pharmaceutical companies in Switzerland", "Food and beverage manufacturers in France"): the hard-filter narrows down to the right country; TF-IDF then selects on industry terms. Precision is high.

- **Industry-specific terminology** ("B2B SaaS companies providing HR solutions in Europe"): NAICS labels and core offering text contain the exact terms the query uses.

- **Equipment manufacturers** ("Renewable energy equipment manufacturers in Scandinavia", "electric vehicle battery production"): NAICS codes like "Battery Manufacturing" or "Turbine and Turbine Generator Set Units Manufacturing" align closely with query intent.

### Where the system struggles

**1. "Logistic companies in Romania"**

There are very few companies in the dataset that use the word "logistics" in their Romanian profiles. The top results include a warehousing company and an industrial supplies merchant, which are tangentially related. The actual logistics companies (freight forwarders, transport operators) may not be in the dataset.

**2. "Fast-growing fintech companies competing with traditional banks in Europe"**

The phrase "fast-growing" and "competing with traditional banks" encode *growth stage* and *competitive positioning*, which TF-IDF cannot infer from static company profiles. Only companies that explicitly use the word "fintech" or "bank" in their descriptions are surfaced. This is where LLM re-ranking helps most.

**3. "E-commerce companies using Shopify or similar platforms"**

"Shopify" is rarely mentioned in company descriptions; instead, most e-commerce companies describe themselves as retailers. The system falls back to matching general e-commerce/retail vocabulary, which has lower precision.

**4. Startup classification** ("Clean energy startups founded after 2018 with fewer than 200 employees")

The word "startup" is not a NAICS concept. The hard filter on `year_founded` and `employee_count` works, but "clean energy" is ambiguous — the dataset contains engineering consultancies that merely mention energy in passing.

---

## Scaling

If the system needed to handle **100,000 companies per query** instead of 477:

| Change | Why |
|---|---|
| **Pre-compute and cache TF-IDF matrix** | Avoid re-transforming corpus on every query (already done — vectorizer fitted once). |
| **Switch to FAISS / approximate nearest neighbour** | Exact cosine similarity over 100k dense vectors becomes a bottleneck; ANN search is orders of magnitude faster. |
| **Use sentence embeddings** | At 100k scale, semantic understanding matters more; embed company profiles offline with a fast model (e.g. `bge-small-en`) and serve queries against a vector index. |
| **Structured filtering via database** | Move hard constraints (country, employee count, revenue, year) to a SQL/Elasticsearch pre-filter; only pass survivors to the embedding/LLM stage. |
| **LLM batch scoring, not per-company** | Keep the batch approach: hard-filter → semantic shortlist (e.g. top 50) → one LLM call with all 50 candidates. |
| **Async parallel LLM calls** | For high-throughput systems, issue multiple batched LLM requests concurrently. |

---

## Failure Modes

### When confident but wrong

1. **Lexical mismatch with correct companies**: A company that is genuinely a logistics operator but uses "supply chain", "freight", "transport" instead of "logistics" may be ranked below an unrelated company that happens to mention "logistic" in a peripheral context.

2. **Geographic false positives**: A company headquartered in France that distributes in Germany could match "German logistics companies" via description text if hard filters are not present.

3. **Country code gaps**: ~15% of companies in this dataset have an address that can't be geocoded (parse errors). These are excluded from country-filtered results even if they would be valid matches.

4. **IDF inflation of rare jargon**: A term like "EV battery" that appears in only 1–2 company profiles gets a very high IDF weight. A company that uses this exact phrase in a generic context (e.g., "we supply components for EV battery production") will score extremely high even if it only supplies a peripheral component.

### What to monitor in production

- **Zero-result queries**: Track queries where the pre-filter eliminates all candidates (overly strict filters) — fall back to semantic-only scoring.
- **Score distribution**: If top results cluster near zero, the query terms are not in the vocabulary (consider query expansion or LLM fallback).
- **False positive rate**: Spot-check a random sample of top results per query with human raters.
- **LLM call latency and cost**: Monitor OpenAI API latency and enforce timeouts; cache LLM results for identical (query, company) pairs.
- **Coverage of country codes**: Track what fraction of companies have parseable addresses; alert if this drops.

---

## Summary

The system combines **structured rule-based filtering** (fast, free, highly accurate for explicit constraints) with **stemmed TF-IDF semantic scoring** (language-variant-robust, zero-cost after indexing) and an optional **LLM re-ranking stage** for semantically complex queries. The design consciously avoids sending every company to an LLM, using the LLM only as a final-stage arbiter over an already-shortlisted set of candidates.

The approach scales to large datasets by keeping the expensive operations (LLM) bounded to a fixed top-K window, while the cheap operations (filtering, TF-IDF) scale linearly with database size and can be offloaded to standard search infrastructure.
