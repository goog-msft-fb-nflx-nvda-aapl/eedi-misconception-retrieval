"""
02_baselines.py
Compute MAP@3 for all baselines:
  1. BM25 (lexical)
  2. TF-IDF + cosine (sparse vector)
  3. Zero-shot E5-small (pre-trained, no fine-tuning)

These are the easiest and fastest to run — all you need for the checkpoint.

Run:  python 02_baselines.py
Output: evaluation/baseline_results.csv  (paste into your report)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.helpers import load_data, build_query, map_at_k, recall_at_k, split_by_question

RESULTS_DIR = Path(__file__).parent / "evaluation"
RESULTS_DIR.mkdir(exist_ok=True)

K_RETRIEVE = 25   # retrieve top-25, then evaluate @3
K_EVAL = 3


# ── 0. Prepare data ────────────────────────────────────────────────────────
def prepare(train_df, misconceptions, val_size=0.15, seed=42):
    """
    Split train into train/val.
    Build:
      - queries: list of query strings
      - labels:  list of correct MisconceptionId (int)
      - corpus:  list of misconception name strings (indexed by position)
      - misc_ids: list of MisconceptionIds (same order as corpus)
    """
    # Build query strings
    train_df["query"] = train_df.apply(build_query, axis=1)

    train_split, val_split = split_by_question(train_df, val_size, seed)

    queries = val_split["query"].tolist()
    labels = val_split["MisconceptionId"].tolist()

    # Corpus = all misconceptions (we retrieve from the full taxonomy)
    corpus = misconceptions["MisconceptionName"].tolist()
    misc_ids = misconceptions["MisconceptionId"].tolist()

    print(f"  Val queries: {len(queries):,}")
    print(f"  Corpus size: {len(corpus):,} misconceptions")
    return queries, labels, corpus, misc_ids


# ── 1. BM25 ───────────────────────────────────────────────────────────────
def run_bm25(queries, labels, corpus, misc_ids):
    from rank_bm25 import BM25Okapi

    print("\n--- BM25 ---")
    t0 = time.time()

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    all_preds = []
    for q in tqdm(queries, desc="BM25"):
        scores = bm25.get_scores(q.lower().split())
        top_idx = np.argsort(scores)[::-1][:K_RETRIEVE]
        all_preds.append([misc_ids[i] for i in top_idx])

    elapsed = time.time() - t0
    score_3 = map_at_k(all_preds, labels, k=3)
    score_r25 = recall_at_k(all_preds, labels, k=25)
    score_25 = map_at_k(all_preds, labels, k=25)
    print(f"  MAP@3:     {score_3:.4f}")
    print(f"  Recall@25: {score_r25:.4f}")
    print(f"  Time:      {elapsed:.1f}s")
    return {"model": "BM25", "MAP@3": score_3, "MAP@25": score_25, "Recall@25": score_r25,
            "time_s": round(elapsed, 1)}


# ── 2. TF-IDF + cosine ────────────────────────────────────────────────────
def run_tfidf(queries, labels, corpus, misc_ids):
    print("\n--- TF-IDF + Cosine ---")
    t0 = time.time()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
    )
    # Fit on corpus + queries together for shared vocab
    all_texts = corpus + queries
    vectorizer.fit(all_texts)

    corpus_vecs = vectorizer.transform(corpus)     # (n_misc, vocab)
    query_vecs = vectorizer.transform(queries)     # (n_queries, vocab)

    # Batch cosine similarity
    all_preds = []
    BATCH = 256
    for i in tqdm(range(0, len(queries), BATCH), desc="TF-IDF"):
        batch_vecs = query_vecs[i:i + BATCH]
        sims = cosine_similarity(batch_vecs, corpus_vecs)  # (batch, n_misc)
        for row_sims in sims:
            top_idx = np.argsort(row_sims)[::-1][:K_RETRIEVE]
            all_preds.append([misc_ids[j] for j in top_idx])

    elapsed = time.time() - t0
    score_3 = map_at_k(all_preds, labels, k=3)
    score_r25 = recall_at_k(all_preds, labels, k=25)
    score_25 = map_at_k(all_preds, labels, k=25)
    print(f"  MAP@3:     {score_3:.4f}")
    print(f"  Recall@25: {score_r25:.4f}")
    print(f"  Time:      {elapsed:.1f}s")
    return {"model": "TF-IDF+cosine", "MAP@3": score_3, "MAP@25": score_25, "Recall@25": score_r25,
            "time_s": round(elapsed, 1)}


# ── 3. Zero-shot E5-small ─────────────────────────────────────────────────
def run_zeroshot_e5(queries, labels, corpus, misc_ids):
    """
    Uses multilingual-e5-small WITHOUT any fine-tuning.
    Requires GPU for reasonable speed (~10 min on CPU, ~1 min on GPU).
    """
    print("\n--- Zero-shot E5-small (multilingual-e5-small) ---")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed — skipping.")
        return None

    t0 = time.time()
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    # E5 requires "query: " / "passage: " prefixes
    q_texts = [f"query: {q}" for q in queries]
    p_texts = [f"passage: {p}" for p in corpus]

    print("  Encoding corpus...")
    corpus_emb = model.encode(
        p_texts, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )
    print("  Encoding queries...")
    query_emb = model.encode(
        q_texts, batch_size=128, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )

    # Cosine sim via dot product (embeddings are L2-normalized)
    print("  Computing similarities...")
    all_preds = []
    BATCH = 512
    for i in tqdm(range(0, len(queries), BATCH), desc="E5 retrieval"):
        batch_emb = query_emb[i:i + BATCH]
        sims = batch_emb @ corpus_emb.T   # (batch, n_misc)
        for row_sims in sims:
            top_idx = np.argsort(row_sims)[::-1][:K_RETRIEVE]
            all_preds.append([misc_ids[j] for j in top_idx])

    elapsed = time.time() - t0
    score_3 = map_at_k(all_preds, labels, k=3)
    score_r25 = recall_at_k(all_preds, labels, k=25)
    score_25 = map_at_k(all_preds, labels, k=25)
    print(f"  MAP@3:     {score_3:.4f}")
    print(f"  Recall@25: {score_r25:.4f}")
    print(f"  Time:      {elapsed:.1f}s")
    return {"model": "Zero-shot E5-small", "MAP@3": score_3, "MAP@25": score_25, "Recall@25": score_r25,
            "time_s": round(elapsed, 1)}


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAG-MAP: Baseline Evaluation")
    print("=" * 60)

    print("\nLoading data...")
    train_df, misconceptions, _ = load_data()

    print("Preparing splits...")
    queries, labels, corpus, misc_ids = prepare(train_df, misconceptions)

    results = []

    # Run all baselines
    r1 = run_bm25(queries, labels, corpus, misc_ids)
    results.append(r1)

    r2 = run_tfidf(queries, labels, corpus, misc_ids)
    results.append(r2)

    r3 = run_zeroshot_e5(queries, labels, corpus, misc_ids)
    if r3:
        results.append(r3)

    # Save results
    df = pd.DataFrame(results)
    from utils.helpers import SPLIT_BY
    out_path = RESULTS_DIR / ("baseline_results.csv" if SPLIT_BY == "QuestionId" else f"baseline_results_{SPLIT_BY}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")
    print("\n" + df.to_string(index=False))
    print("\n→ Copy this table into your checkpoint report (Section 4 / Evaluation).")


if __name__ == "__main__":
    main()
