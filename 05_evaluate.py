"""
05_evaluate.py
Full ablation study: compare all RAG-MAP components on val set.

Ablation table:
  BM25                     → MAP@3
  TF-IDF + cosine          → MAP@3
  Zero-shot E5-small       → MAP@3
  Fine-tuned E5-small      → MAP@3   (retriever only)
  Fine-tuned E5 + reranker → MAP@3   (retriever + reranker)

Also produces:
  - evaluation/ablation_table.csv
  - evaluation/ablation_plot.png
  - evaluation/error_analysis.csv  (worst-performing queries)

Run:  python 05_evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.helpers import load_data, build_query, map_at_k, recall_at_k, MODELS_DIR

EVAL_DIR = Path(__file__).parent / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)
RETRIEVER_DIR = MODELS_DIR / "retriever"
RERANKER_DIR  = MODELS_DIR / "reranker"

VAL_SIZE = 0.15
SEED = 42
K_RETRIEVE = 25
K_EVAL = 3


# ── Retrieval helpers ─────────────────────────────────────────────────────
def retrieve_bm25(queries, corpus, misc_ids, k=25):
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([d.lower().split() for d in corpus])
    all_preds = []
    for q in tqdm(queries, desc="BM25", leave=False):
        scores = bm25.get_scores(q.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        all_preds.append([misc_ids[i] for i in top_idx])
    return all_preds


def retrieve_tfidf(queries, corpus, misc_ids, k=25):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
    vec.fit(corpus + queries)
    corp_v = vec.transform(corpus)
    q_v = vec.transform(queries)
    sims = cosine_similarity(q_v, corp_v)
    all_preds = []
    for row in sims:
        top_idx = np.argsort(row)[::-1][:k]
        all_preds.append([misc_ids[i] for i in top_idx])
    return all_preds


def retrieve_biencoder(queries, corpus, misc_ids, model_name_or_path, k=25):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name_or_path)
    corp_emb = model.encode(
        [f"passage: {c}" for c in corpus], batch_size=256,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    )
    q_emb = model.encode(
        [f"query: {q}" for q in queries], batch_size=128,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    )
    all_preds = []
    BATCH = 512
    for i in range(0, len(queries), BATCH):
        sims = q_emb[i:i + BATCH] @ corp_emb.T
        for row in sims:
            top_idx = np.argsort(row)[::-1][:k]
            all_preds.append([misc_ids[j] for j in top_idx])
    return all_preds


def rerank(queries, retrieved_preds, corpus, misc_ids, reranker_path, k=25):
    """Apply cross-encoder reranking on top of retrieved candidates."""
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        reranker = CrossEncoder(reranker_path)
    except Exception:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained(reranker_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            reranker_path, num_labels=1
        )
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        misc_id_to_name = dict(zip(misc_ids, corpus))
        all_reranked = []
        for q, cand_ids in tqdm(zip(queries, retrieved_preds),
                                 total=len(queries), desc="Reranking"):
            cand_names = [misc_id_to_name.get(cid, "") for cid in cand_ids]
            pairs = [(q, cn) for cn in cand_names]
            enc = tokenizer(
                [p[0] for p in pairs], [p[1] for p in pairs],
                padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                scores = model(**enc).logits.squeeze(-1).cpu().numpy()
            reranked_order = np.argsort(scores)[::-1]
            all_reranked.append([cand_ids[i] for i in reranked_order])
        return all_reranked

    # sentence-transformers CrossEncoder path
    misc_id_to_name = dict(zip(misc_ids, corpus))
    all_reranked = []
    for q, cand_ids in tqdm(zip(queries, retrieved_preds),
                             total=len(queries), desc="Reranking"):
        cand_names = [misc_id_to_name.get(cid, "") for cid in cand_ids]
        pairs = [(q, cn) for cn in cand_names]
        scores = reranker.predict(pairs, show_progress_bar=False)
        reranked_order = np.argsort(scores)[::-1]
        all_reranked.append([cand_ids[i] for i in reranked_order])
    return all_reranked


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAG-MAP: Full Ablation Study")
    print("=" * 60)

    train_df, misconceptions, _ = load_data()
    train_df["query"] = train_df.apply(build_query, axis=1)
    _, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED)

    queries = val_df["query"].tolist()
    labels = val_df["MisconceptionId"].tolist()
    corpus = misconceptions["MisconceptionName"].tolist()
    misc_ids = misconceptions["MisconceptionId"].tolist()

    results = []

    # ── BM25 ──
    print("\n[1/5] BM25...")
    preds_bm25 = retrieve_bm25(queries, corpus, misc_ids)
    results.append({
        "Model": "BM25 (lexical)",
        "MAP@3": map_at_k(preds_bm25, labels, 3),
        "Recall@25": recall_at_k(preds_bm25, labels, 25),
    })

    # ── TF-IDF ──
    print("\n[2/5] TF-IDF + cosine...")
    preds_tfidf = retrieve_tfidf(queries, corpus, misc_ids)
    results.append({
        "Model": "TF-IDF + cosine",
        "MAP@3": map_at_k(preds_tfidf, labels, 3),
        "Recall@25": recall_at_k(preds_tfidf, labels, 25),
    })

    # ── Zero-shot E5 ──
    print("\n[3/5] Zero-shot E5-small...")
    preds_e5_zs = retrieve_biencoder(
        queries, corpus, misc_ids, "intfloat/multilingual-e5-small"
    )
    results.append({
        "Model": "Zero-shot E5-small",
        "MAP@3": map_at_k(preds_e5_zs, labels, 3),
        "Recall@25": recall_at_k(preds_e5_zs, labels, 25),
    })

    # ── Fine-tuned E5 ──
    if RETRIEVER_DIR.exists():
        print("\n[4/5] Fine-tuned E5-small (retriever only)...")
        preds_e5_ft = retrieve_biencoder(queries, corpus, misc_ids, str(RETRIEVER_DIR))
        results.append({
            "Model": "Fine-tuned E5-small",
            "MAP@3": map_at_k(preds_e5_ft, labels, 3),
            "Recall@25": recall_at_k(preds_e5_ft, labels, 25),
        })

        # ── Fine-tuned E5 + Reranker ──
        if RERANKER_DIR.exists():
            print("\n[5/5] Fine-tuned E5 + Cross-encoder Reranker...")
            preds_reranked = rerank(
                queries, preds_e5_ft, corpus, misc_ids, str(RERANKER_DIR)
            )
            results.append({
                "Model": "E5 + Cross-encoder Reranker",
                "MAP@3": map_at_k(preds_reranked, labels, 3),
                "Recall@25": recall_at_k(preds_reranked, labels, 25),
            })
        else:
            print("\n[5/5] Reranker model not found — skipping. "
                  "Run 04_train_reranker.py first.")
    else:
        print("\n[4/5] Retriever checkpoint not found. Run 03_train_retriever.py first.")

    # ── Save table ────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df["MAP@3"] = df["MAP@3"].round(4)
    df["Recall@25"] = df["Recall@25"].round(4)

    csv_path = EVAL_DIR / "ablation_table.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 50)
    print("ABLATION RESULTS")
    print("=" * 50)
    print(df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette("Blues_d", len(df))

    for ax, metric in zip(axes, ["MAP@3", "Recall@25"]):
        bars = ax.barh(df["Model"], df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} — Ablation Study")
        ax.invert_yaxis()
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

    fig.suptitle("RAG-MAP Ablation Study", fontweight="bold")
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "ablation_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Results → {csv_path}")
    print(f"✓ Plot    → {EVAL_DIR / 'ablation_plot.png'}")
    print("\n→ This table goes into Section 4 of your checkpoint report.")


if __name__ == "__main__":
    main()
