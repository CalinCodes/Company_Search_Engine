# WRITEUP: Intent Qualification System (VibeHack 2026)

## 1. Approach (System Architecture)
The system is designed as a **multistage funnel pipeline**, intended to filter a large volume of raw data and deliver qualified results through deep semantic analysis in the final stage.

### Components and Interaction:
1.  **Stage 1: Intent Deconstruction (The Parser)**: 
    * Uses the **DeepSeek-V3** model to transform a natural language query into a structured JSON object.
    * Extracts "hard" filters (countries, employee count, revenue, NAICS codes) and **semantic keywords**.
    * Identifies the **business role** of the target (e.g., Supplier, Competitor, Manufacturer) to refine the search context.
2.  **Stage 1b: Deterministic Filtering**:
    * Applies the extracted filters directly to the dataset (`final_processed_data.json`). This stage drastically reduces the search space using SQL-like criteria, eliminating companies that do not meet basic constraints (e.g., location or size).
3.  **Stage 2: Hybrid Retrieval (The Ranker)**:
    * Ranks the remaining candidates using a combined score: **BM25** (for exact lexical matches) + **Cosine Similarity** on embeddings (`all-MiniLM-L6-v2`).
    * This method ensures that queries containing specific technical terms are captured as well as those based on abstract concepts. 
4.  **Stage 3: LLM Final Filter (The Judge)**:
    * The best `top_k` candidates are sent to **Qwen2.5-7B-Instruct**.
    * The model receives the full company context (description, business model, target markets) and decides if it *truly* satisfies the user's intent, providing a reasoning sentence.

---

## 2. Tradeoffs (Optimizations)
* **Accuracy vs. Cost:** Instead of sending 500 companies directly to an LLM (expensive and slow), we use Stage 1 and 2 to reduce the list to the most relevant ~10-20 candidates.
* **Speed vs. Understanding:** By using local embedding models and the `rank-bm25` library, the ranking stage is extremely fast. The LLM is reserved only for the final "judgment-heavy" decision.
* **Robustness to Missing Data:** The pipeline includes a **Relaxed Query** function. If Stage 1 returns 0 results due to overly strict filters, the system automatically removes constraints (such as revenue) and retries the search based on semantic context alone.

---

## 3. Error Analysis
* **Where the system struggles:**
    * **NAICS Ambiguity:** Some companies have generic NAICS codes. If the query is very specific, Stage 1 might not find a perfect NAICS code, relying heavily on Stage 2 (semantic).
    * **Multinational Entities:** A company might be registered in Germany but have production activities only in Asia. The country filter (Stage 1) might keep it, even though it doesn't satisfy a "local production" intent.
* **Misclassification Example:** A logistics software provider might pass Stage 2 for the query "Logistic companies in Romania" because the description contains the word "logistics". Stage 3 (Qwen) is designed to identify this difference in role, but if the description is vague, the LLM might produce a false positive.

---

## 4. Scaling
If the system needed to handle **100,000 companies** per query:
1.  **Vector Database:** Replace in-memory filtering with a vector database (e.g., Qdrant, Pinecone, or Milvus) for Stage 2.
2.  **Distributed Processing:** Batch the Stage 3 (LLM) process across multiple GPU instances in parallel.
3.  **Pre-indexing:** Calculate embeddings offline (asynchronously) for the entire dataset, rather than at query time.

---

## 5. Failure Modes and Monitoring
* **Hallucination in Parsing:** Stage 1 might incorrectly deduce a NAICS code or a country. 
    * *Solution:* Monitor the `reasoning` provided by the Parser and log cases where `total_output` in Stage 1 is 0.
* **API Latency:** Dependence on external APIs (Featherless, Stripe) can stall the user experience.
    * *Monitoring:* Dashboards for API latency and rate-limiting alerts.

---

## 6. Critical Thinking
* **Key Signals:** The system relies heavily on the quality of the `description` and `core_offerings` fields. If these are missing, Stage 2 and 3 become less effective.
* **Multilingual Support:** The implementation includes a translation layer (Google Translate API) that allows users to search in their native language while running the pipeline in English for consistency.

---
