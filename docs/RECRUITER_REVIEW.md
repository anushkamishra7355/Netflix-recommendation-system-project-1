# Recruiter Review — Netflix Recommendation System

*Reviewed from the perspective of an Amazon ML / SDE technical recruiter.*

---

## Strengths

1. **End-to-end ML pipeline** — Demonstrates complete workflow from raw data → preprocessing → feature engineering → model artifacts → deployed web application.

2. **Correct ML fundamentals** — Uses industry-standard content-based filtering with TF-IDF and cosine similarity; concepts are well-documented with mathematical notation.

3. **Production-minded refactoring** — Fixed critical indexing bug, eliminated 510 MB matrix recomputation, externalized API keys, and added deployment configs for multiple platforms.

4. **Clean code organization** — Separated concerns into `src/recommender.py`, `src/poster.py`, `src/evaluation.py`, `app.py`, and `scripts/`.

5. **Reproducibility** — `train_model.py` enables anyone to regenerate model files from the dataset; notebook preserved for EDA.

6. **User experience** — Streamlit UI includes loading spinners, error handling, similarity scores, and optional movie posters.

7. **Offline evaluation** — Genre-overlap proxy metrics (precision@K, recall@K, F1@K, hit rate@K) with documented methodology in `docs/evaluation_results.md`.

8. **Documentation depth** — ML documentation, architecture diagrams, Amazon MLSS write-up, and scalability analysis show engineering maturity beyond a tutorial project.

---

## Weaknesses

1. **No automated tests** — Missing unit tests for `recommend()` function, preprocessing, or integration tests for the Streamlit app.

2. **Proxy-based evaluation only** — Genre overlap is a reasonable proxy, but there is no ground-truth user feedback or qualitative user study.

3. **Basic ML approach** — TF-IDF bag-of-words is introductory; no embeddings, matrix factorization, or hybrid methods.

4. **No CI/CD pipeline** — No GitHub Actions for linting, testing, or automated deployment.

5. **Large dataset in repo** — 3.4 MB CSV committed; could use DVC or download script instead.

6. **Limited personalization** — Pure content-based; cannot adapt to individual user preferences or viewing history.

7. **No monitoring/logging** — Production systems need request logging, latency metrics, and error tracking.

---

## Improvements Needed

| Priority | Improvement |
|----------|-------------|
| **High** | Add pytest unit tests for recommender and preprocessing |
| **Medium** | Add GitHub Actions CI for test + lint |
| **Medium** | Upgrade to Sentence-BERT embeddings for semantic similarity |
| **Medium** | Add `pyproject.toml` or `setup.cfg` for package management |
| **Low** | Add Docker containerization |
| **Low** | Implement recommendation explanation (show shared keywords) |
| **Low** | Collect user feedback or click-through data for stronger evaluation |

---

## Resume Bullet Points

Use these on your resume (customize with your name/metrics):

- **Built a content-based Netflix recommendation engine** serving 8,800+ titles using TF-IDF vectorization and cosine similarity, deployed as an interactive Streamlit web application.

- **Engineered end-to-end ML pipeline** including data cleaning, feature engineering (description + cast + director), sparse matrix optimization, and artifact serialization reducing model load time by 10x.

- **Deployed ML application to cloud platforms** (Streamlit Cloud, Render, Railway) with environment-based configuration, automated artifact builds, and production error handling.

- **Implemented offline evaluation** achieving 51.7% precision@10 and 96.3% hit rate@10 using genre-overlap relevance on 8,807 titles.

- **Diagnosed and fixed critical indexing bug** in similarity computation that caused incorrect recommendations; refactored monolithic script into modular, testable Python package.

- **Authored comprehensive ML documentation** covering mathematical foundations (TF-IDF, cosine similarity), architectural decisions, and scalability roadmap for 100K+ title catalogs.

---

## Likely Interview Questions

### Machine Learning

1. **Explain TF-IDF. Why use it instead of raw word counts?**
   - Expected: Discuss term frequency, inverse document frequency, down-weighting common terms.

2. **Why cosine similarity for text vectors?**
   - Expected: Magnitude-invariant, works well with sparse high-dimensional vectors, range [0,1].

3. **What are the limitations of content-based filtering?**
   - Expected: No personalization, overspecialization, cold-start for new items with sparse metadata.

4. **How did you evaluate this recommender without user feedback?**
   - Expected: Genre overlap proxy, precision@K / hit rate@K, manual inspection of top recommendations, discussion of recall limitations when relevant sets are large.

5. **How would you scale this to 1 million titles?**
   - Expected: ANN (FAISS), embedding models, precomputed top-K, sharding, caching.

### System Design

6. **Walk me through your deployment architecture.**
   - Expected: Build artifacts offline, load at startup with caching, row-wise inference, optional poster API.

7. **What bug did you find and how did you fix it?**
   - Expected: Index mismatch between DataFrame labels and matrix rows; fixed with positional mapping.

8. **How do you handle the OMDb API key securely?**
   - Expected: Environment variables, Streamlit secrets, never commit to git.

### Coding

9. **Write a function to return top-K similar titles given a query vector.**
   - Expected: Cosine similarity, argsort, exclude self, return top K.

10. **How would you add unit tests for the recommender?**
    - Expected: Known similar pairs (same director), unknown title raises KeyError, output length equals top_n.

### Behavioral

11. **Why did you choose this project for Amazon MLSS?**
    - Prepare: Connect to interest in recommendation systems, NLP, and production ML.

12. **What would you do differently with more time?**
    - Prepare: Embeddings, user-feedback evaluation, A/B testing, hybrid model.

---

*Use this document to prepare for Amazon ML Summer School and SDE interviews.*
