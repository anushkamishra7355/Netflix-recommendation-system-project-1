# Amazon ML Summer School — Project Documentation

## Netflix Content-Based Recommendation System

**Author:** Anushka Mishra  
**Repository:** [github.com/anushkamishra7355/netflix-recommendation-system](https://github.com/anushkamishra7355/netflix-recommendation-system)

---

## Project Summary

This project builds a deployable **content-based recommender** for Netflix titles. Given a movie or TV show, the system returns the most similar titles in the catalog using TF-IDF text features and cosine similarity. The solution includes:

- Reproducible training pipeline (`scripts/train_model.py`)
- Cached inference in a Streamlit web app (`app.py`)
- Offline evaluation without user ratings (`scripts/evaluate_model.py`)
- Deployment configs for Streamlit Cloud, Render, and Railway

The project demonstrates classical NLP + information retrieval techniques applied to a real catalog-recommendation problem, with attention to correctness, deployment, and measurable quality.

---

## Machine Learning Concepts Used

| Concept | Application in Project |
|---------|-------------------------|
| **Content-Based Filtering** | Recommendations derived from item features (text metadata), not user behavior |
| **Bag-of-Words (BoW)** | Each title represented as a collection of word tokens from combined text |
| **TF-IDF Vectorization** | Converts text to weighted numerical features emphasizing discriminative terms |
| **Cosine Similarity** | Measures angular similarity between title vectors in high-dimensional space |
| **Sparse Matrix Operations** | Efficient storage and computation for high-dimensional, mostly-zero TF-IDF vectors |
| **Top-K Retrieval** | Returns the K highest-scoring similar titles for a query |
| **Offline Evaluation** | Genre-overlap proxy metrics when explicit user feedback is unavailable |

---

## Mathematical Concepts Used

### 1. Term Frequency (TF)
\[
\text{TF}(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in } d}
\]

### 2. Inverse Document Frequency (IDF)
\[
\text{IDF}(t) = \log \frac{N}{1 + \text{df}(t)}
\]

### 3. TF-IDF Weight
\[
w_{t,d} = \text{TF}(t,d) \times \text{IDF}(t)
\]

### 4. Cosine Similarity
\[
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2} = \frac{\sum_i A_i B_i}{\sqrt{\sum_i A_i^2} \sqrt{\sum_i B_i^2}}
\]

### 5. N-gram Language Model
Bigrams and trigrams capture local word dependencies:
- Unigram: `"action"`
- Bigram: `"action movie"`
- Trigram: `"action movie hero"`

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.11+ | Core language |
| **pandas** | 2.2.3 | Data loading, cleaning, manipulation |
| **scikit-learn** | 1.5.2 | TfidfVectorizer, cosine_similarity |
| **scipy** | 1.14.1 | Sparse matrix backend |
| **numpy** | 2.1.3 | Numerical operations |
| **Streamlit** | 1.40.1 | Interactive web UI |
| **requests** | 2.32.3 | OMDb API poster fetching |

---

## Model Workflow

```mermaid
flowchart TB
    subgraph Training["Offline Training (train_model.py)"]
        A1[Load data/netflix_titles.csv] --> A2[Clean & impute missing values]
        A2 --> A3[Create combined text feature]
        A3 --> A4[Fit TfidfVectorizer]
        A4 --> A5[Transform to sparse TF-IDF matrix]
        A5 --> A6[Save titles.pkl and tfidf_matrix.pkl]
    end

    subgraph Inference["Online Inference (app.py)"]
        B1[Load artifacts with @st.cache_resource] --> B2[User selects title]
        B2 --> B3[Compute cosine similarity for query row]
        B3 --> B4[Sort and return top-N titles]
        B4 --> B5[Fetch posters from OMDb API]
        B5 --> B6[Render results in Streamlit]
    end

    subgraph Evaluation["Offline Evaluation (evaluate_model.py)"]
        C1[Genre overlap relevance proxy] --> C2[Precision@K / Recall@K / F1@K]
        C2 --> C3[Hit Rate@K across 8,807 titles]
    end

    Training --> Inference
    Training --> Evaluation
```

### Training Phase
1. Load raw CSV from `data/netflix_titles.csv`
2. Apply preprocessing pipeline
3. Fit TF-IDF vectorizer on all `combined` text
4. Persist runtime artifacts to `models/`:
   - `titles.pkl` — cleaned title metadata
   - `tfidf_matrix.pkl` — sparse vectors used at inference
   - `vectorizer.pkl` — saved for reproducibility; not loaded by the app

### Inference Phase
1. Load cached recommender on app startup
2. On user query: compute one row of cosine similarity
3. Return ranked recommendations with scores

### Evaluation Phase
Because the dataset has no user watch history, evaluation uses **genre overlap** (`listed_in`) as a relevance proxy:

| Metric | @K = 10 | Interpretation |
|--------|--------:|----------------|
| Precision@K | 0.5169 | About half of top-10 recommendations share a genre with the query |
| Recall@K | 0.0039 | Low because the relevant genre set per title is large |
| F1@K | 0.0076 | Harmonic mean of precision and recall |
| Hit Rate@K | 0.9633 | 96.3% of queries receive at least one genre-matching recommendation |

Full methodology: [evaluation_results.md](evaluation_results.md)

---

## Technical Challenges

| Challenge | Solution |
|-----------|----------|
| **Index mismatch bug** | Original code used non-contiguous DataFrame indices to access similarity matrix rows, producing incorrect recommendations. Fixed with positional `title_to_idx` mapping. |
| **510 MB similarity matrix** | Full pairwise matrix too large for Git/deployment. Switched to on-demand row-wise cosine similarity (~3.4 MB sparse matrix). |
| **Slow cold start** | Original app recomputed full sigmoid kernel on every startup. Precomputed TF-IDF artifacts reduce load time to ~1–2 seconds. |
| **Hardcoded API key** | Moved OMDb key to environment variable `OMDB_API_KEY` for security. |
| **Inconsistent vectorizer params** | App used different TF-IDF settings than notebook. Unified parameters in `build_vectorizer()`. |
| **Missing deployment configs** | Added Procfile, render.yaml, railway.json, and `.streamlit/config.toml`. |
| **No explicit labels for evaluation** | Implemented genre-overlap proxy metrics and documented limitations. |

---

## Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Content-based over collaborative** | No user rating data available; metadata is rich and publicly accessible |
| **TF-IDF over deep learning** | Interpretable, fast, no GPU, suitable for portfolio/demo deployment |
| **Cosine over sigmoid kernel** | Standard for text similarity; mathematically equivalent ranking for normalized vectors |
| **Precomputed artifacts** | Separates training from inference; enables reproducible builds via `train_model.py` |
| **Modular src/ package** | `recommender.py`, `poster.py`, and `evaluation.py` separated from UI for testability |
| **Streamlit for UI** | Rapid prototyping, free cloud deployment, Python-native |
| **Row-wise similarity at inference** | Avoids storing an N×N matrix; only `titles.pkl` + `tfidf_matrix.pkl` required at runtime |

---

## Scalability Considerations

### Current Scale (~8.8K titles)
- TF-IDF matrix: ~3.4 MB sparse
- Inference: O(N) per query via single-row cosine similarity
- Memory: ~50 MB loaded in Streamlit process

### Medium Scale (~100K titles)
- **Approximate Nearest Neighbors (ANN)** — FAISS, Annoy, or HNSW for sub-linear search
- **Batch precomputation** — Precompute top-K neighbors offline, serve from lookup table
- **Feature hashing** — Reduce vocabulary dimension with HashingVectorizer

### Large Scale (~1M+ titles)
- **Distributed indexing** — Shard catalog by genre/country
- **Embedding models** — Sentence-BERT with vector database (Pinecone, Milvus)
- **Caching layer** — Redis for popular query results
- **Microservice architecture** — Separate recommendation API from UI
- **Model refresh pipeline** — Airflow/cron for nightly artifact rebuilds

### Deployment Scalability

| Platform | Artifact strategy | Entry point |
|----------|-------------------|-------------|
| **Streamlit Cloud** | Commit `models/*.pkl` or build locally before deploy | `app.py` |
| **Render** | `render.yaml` runs `train_model.py` during build | `app.py` |
| **Railway** | `railway.json` runs `train_model.py` during build | `app.py` |
| **Heroku / Procfile** | `setup.sh` trains if `models/titles.pkl` is missing | `app.py` |

---

## Why This Project Fits Amazon MLSS

1. **Classical ML + NLP fundamentals** — TF-IDF, sparse linear algebra, and similarity search are core retrieval concepts.
2. **End-to-end systems thinking** — Training, inference, evaluation, and deployment are separated cleanly.
3. **Correctness focus** — A real indexing bug was identified and fixed with positional lookups.
4. **Measurable outcomes** — Offline metrics and example recommendations demonstrate system behavior.
5. **Production awareness** — Secrets management, artifact builds, and multi-platform deploy configs are included.

---

## Suggested Talking Points for Review

- Why content-based filtering is appropriate when only item metadata exists
- How cosine similarity on sparse vectors enables fast top-K retrieval
- Why a full precomputed similarity matrix was replaced with row-wise computation
- How genre-overlap evaluation works and what its limitations are
- How the system would evolve with embeddings, ANN indexing, and user feedback

---

*Prepared for Amazon Machine Learning Summer School application.*
