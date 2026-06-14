# Machine Learning Documentation

## Netflix Content-Based Recommendation System

---

## 1. Problem Statement

Streaming platforms like Netflix offer thousands of movies and TV shows. Users often struggle to discover content aligned with their preferences. This project solves a **content-based recommendation** problem: given a title the user likes, recommend similar titles using metadata and text descriptions—without requiring user rating history or collaborative filtering data.

**Objective:** Build an interpretable, deployable recommender that maps each title to semantically similar content based on description, cast, and director information.

---

## 2. Dataset Used

| Property | Value |
|----------|-------|
| **Source** | [Netflix Titles Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Kaggle) |
| **File** | `data/netflix_titles.csv` |
| **Records** | 8,807 titles |
| **Features** | `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description` |

The dataset spans movies and TV shows across multiple countries, genres, and release years.

---

## 3. Data Cleaning

Cleaning steps applied in `scripts/train_model.py` and documented in `notebooks/netflix-recommendation-system.ipynb`:

1. **Missing country values** — Imputed with `"United States"` (most frequent country in the dataset).
2. **Missing text fields** — `description`, `cast`, and `director` filled with empty strings before concatenation.
3. **Duplicate titles** — Removed duplicate `title` entries (keeping the first occurrence) to ensure one recommendation index per title.
4. **Index reset** — DataFrame re-indexed to contiguous positional indices for reliable matrix lookups.

Exploratory analysis in the notebook also examined null counts, country distributions, and genre frequency before modeling.

---

## 4. Feature Engineering

A single composite text feature **`combined`** is created by concatenating:

```
combined = description + " " + cast + " " + director
```

**Rationale:**
- **Description** captures plot themes and tone.
- **Cast** links titles with shared actors (franchise/ensemble similarity).
- **Director** connects stylistically similar works by the same creator.

This bag-of-words representation treats each title as a document, enabling text similarity methods without manual feature design.

---

## 5. Text Preprocessing

Preprocessing is handled inside scikit-learn's `TfidfVectorizer`:

| Step | Implementation |
|------|----------------|
| Tokenization | Word-level analyzer |
| Stop word removal | English stop words (`stop_words='english'`) |
| N-grams | Unigrams, bigrams, and trigrams `(1, 3)` |
| Minimum document frequency | `min_df=3` (ignore terms appearing in fewer than 3 titles) |
| Case normalization | Lowercasing (default) |

No stemming or lemmatization is applied; n-grams partially compensate by preserving phrase context.

---

## 6. TF-IDF Vectorization

**Term Frequency–Inverse Document Frequency (TF-IDF)** converts each title's `combined` text into a sparse numerical vector.

For term \( t \) in document \( d \):

\[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
\]

\[
\text{IDF}(t) = \log \frac{N}{1 + \text{df}(t)}
\]

Where:
- \( N \) = total number of titles
- \( \text{df}(t) \) = number of titles containing term \( t \)

**Output:** Sparse matrix of shape `(8807, vocabulary_size)` saved as `models/tfidf_matrix.pkl`.

TF-IDF down-weights common words (e.g., "the", "movie") and up-weights discriminative terms (e.g., specific actor names, plot keywords).

---

## 7. Cosine Similarity

Similarity between titles \( i \) and \( j \) is computed as **cosine similarity** between their TF-IDF vectors:

\[
\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
\]

Cosine similarity ranges from 0 (orthogonal/no shared terms) to 1 (identical term profiles).

**Why cosine over Euclidean distance?** Cosine measures orientation (content overlap) rather than magnitude, making it ideal for sparse, high-dimensional text vectors.

At inference time, only the query title's row is compared against the full matrix—avoiding storage of an \( N \times N \) similarity matrix (~510 MB for 8,807 titles).

---

## 8. Recommendation Pipeline

```mermaid
flowchart LR
    A[User selects title] --> B[Lookup positional index]
    B --> C[Extract TF-IDF vector]
    C --> D[Compute cosine similarity vs all titles]
    D --> E[Rank by score descending]
    E --> F[Return top-N excluding query]
    F --> G[Fetch posters via OMDb API]
    G --> H[Display in Streamlit UI]
```

**Steps:**
1. User selects a title from the dropdown in the Streamlit app.
2. System maps title → positional index via `title_to_idx` dictionary.
3. Cosine similarity computed between query vector and all title vectors.
4. Top 10 titles (excluding the query) returned with similarity scores.
5. Optional: OMDb API fetches movie posters for visual display.

---

## 9. Advantages

| Advantage | Description |
|-----------|-------------|
| **No cold-start for new users** | Recommendations work from content alone; no rating history needed |
| **Interpretable** | Similarity driven by shared cast, director, and description terms |
| **Fast inference** | Row-wise cosine similarity on sparse matrices (~milliseconds) |
| **Lightweight deployment** | Artifacts ~5 MB; no GPU required |
| **Scalable startup** | Precomputed TF-IDF matrix avoids recomputation on each app load |

---

## 10. Limitations

| Limitation | Impact |
|------------|--------|
| **No user personalization** | Same query always returns identical results |
| **Metadata quality dependent** | Poor or missing descriptions reduce recommendation quality |
| **Lexical similarity only** | Cannot capture semantic meaning beyond shared words (e.g., "happy" vs "joyful") |
| **Popularity bias absent** | Cannot boost trending or highly-rated content |
| **Duplicate franchise entries** | Sequels/remakes with different titles may not cluster well |
| **Poster API dependency** | OMDb requires API key; posters may be unavailable |

---

## 11. Offline Evaluation

Because the dataset has no user ratings or watch history, recommendation quality is measured with a **genre-overlap proxy**:

- **Relevant items:** titles sharing at least one `listed_in` category with the query title
- **Precision@K:** fraction of top-K recommendations that share a genre
- **Recall@K:** fraction of all genre-relevant titles retrieved in top-K
- **F1@K:** harmonic mean of precision and recall
- **Hit Rate@K:** fraction of queries with at least one genre-matching recommendation

### Results (@K = 10)

| Metric | Score |
|--------|------:|
| Precision@K | 0.5169 |
| Recall@K | 0.0039 |
| F1@K | 0.0076 |
| Hit Rate@K | 0.9633 |

Run `python scripts/evaluate_model.py` to regenerate metrics. See [evaluation_results.md](evaluation_results.md) for methodology and interpretation.

---

## 12. Runtime Artifacts

| File | Required at inference? | Purpose |
|------|----------------------|---------|
| `models/titles.pkl` | Yes | Cleaned title metadata and lookup table |
| `models/tfidf_matrix.pkl` | Yes | Sparse TF-IDF vectors for cosine similarity |
| `models/vectorizer.pkl` | No | Saved during training for reproducibility |

---

## 13. Future Scope

1. **Sentence embeddings** — Replace TF-IDF with BERT/Sentence-BERT for semantic similarity.
2. **Hybrid model** — Combine content-based scores with collaborative filtering from user ratings.
3. **Feature enrichment** — Add `listed_in` (genres), `rating`, and `release_year` as weighted features.
4. **Approximate nearest neighbors** — Use FAISS or Annoy for sub-linear search at million-title scale.
5. **A/B testing framework** — Measure click-through rate on recommendations in production.
6. **Explainability** — Highlight shared terms (cast, keywords) driving each recommendation.
7. **Real-time updates** — Incremental TF-IDF updates as new titles are added to the catalog.
8. **Automated tests and CI** — Add pytest coverage and GitHub Actions for regression safety.

---

*Generated for Amazon ML Summer School application portfolio.*
