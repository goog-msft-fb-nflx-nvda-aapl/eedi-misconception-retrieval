"""
01_eda.py
Exploratory Data Analysis for the MAP / Eedi dataset.

Produces:
  eda/figures/01_label_distribution.png
  eda/figures/02_response_length.png
  eda/figures/03_top25_misconceptions.png
  eda/figures/04_subject_distribution.png
  eda/figures/05_wordcloud_questions.png
  eda/figures/06_vocab_overlap.png
  eda/eda_summary.txt   ← copy key numbers into your report

Run:  python 01_eda.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

from utils.helpers import load_data, build_query, FIG_DIR

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (10, 5)


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def run_eda():
    print("=" * 60)
    print("RAG-MAP: Exploratory Data Analysis")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    train_df, misconceptions, test_df = load_data()
    n_train = len(train_df)
    n_misconceptions = len(misconceptions)

    print(f"  Training pairs (distractor–misconception): {n_train:,}")
    print(f"  Unique questions:  {train_df['QuestionId'].nunique():,}")
    print(f"  Unique misconceptions in train: {train_df['MisconceptionId'].nunique():,}")
    print(f"  Total misconception categories: {n_misconceptions:,}")
    print(f"  Test rows: {len(test_df):,}")

    # ── 1. Label distribution ──────────────────────────────────────
    print("\n[2/7] Label distribution...")
    label_counts = train_df["MisconceptionId"].value_counts()
    gini = 1 - sum((c / n_train) ** 2 for c in label_counts.values)

    print(f"  Most common misconception count:  {label_counts.iloc[0]:,}")
    print(f"  Least common misconception count: {label_counts.iloc[-1]:,}")
    print(f"  Gini impurity (imbalance): {gini:.4f}")
    print(f"  Misconceptions seen ≥5 times: {(label_counts >= 5).sum():,}")
    print(f"  Misconceptions seen only once: {(label_counts == 1).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(label_counts.values, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Samples per Misconception")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Label Frequency Distribution")

    # CDF
    sorted_counts = np.sort(label_counts.values)[::-1]
    cdf = np.cumsum(sorted_counts) / sorted_counts.sum()
    axes[1].plot(range(1, len(sorted_counts) + 1), cdf, color="tomato")
    axes[1].axhline(0.8, linestyle="--", color="gray", label="80% coverage")
    axes[1].set_xlabel("Number of Misconception Classes (ranked)")
    axes[1].set_ylabel("Cumulative Coverage")
    axes[1].set_title("Coverage CDF — Label Imbalance")
    axes[1].legend()
    fig.suptitle("Label Distribution Analysis", fontweight="bold")
    save(fig, "01_label_distribution.png")

    # ── 2. Text length distributions ──────────────────────────────
    print("\n[3/7] Text length distributions...")
    train_df["q_len"] = train_df["QuestionText"].fillna("").str.split().str.len()
    train_df["wrong_len"] = train_df["WrongAnswerText"].fillna("").str.split().str.len()
    train_df["correct_len"] = train_df["CorrectAnswerText"].fillna("").str.split().str.len()
    misconceptions["m_len"] = misconceptions["MisconceptionName"].fillna("").str.split().str.len()

    print(f"  Question length   — mean: {train_df['q_len'].mean():.1f}  "
          f"median: {train_df['q_len'].median():.0f}  max: {train_df['q_len'].max()}")
    print(f"  Wrong answer len  — mean: {train_df['wrong_len'].mean():.1f}  "
          f"median: {train_df['wrong_len'].median():.0f}")
    print(f"  Misconception len — mean: {misconceptions['m_len'].mean():.1f}  "
          f"median: {misconceptions['m_len'].median():.0f}  max: {misconceptions['m_len'].max()}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, label in zip(
        axes,
        ["q_len", "wrong_len"],
        ["Question Text", "Wrong Answer Text"],
    ):
        ax.hist(train_df[col], bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label} Length")

    axes[2].hist(misconceptions["m_len"], bins=30, color="mediumseagreen", edgecolor="white")
    axes[2].set_xlabel("Word Count")
    axes[2].set_title("Misconception Name Length")
    fig.suptitle("Text Length Distributions", fontweight="bold")
    save(fig, "02_response_length.png")

    # ── 3. Top-25 misconceptions ───────────────────────────────────
    print("\n[4/7] Top misconceptions...")
    top25 = (
        train_df.groupby(["MisconceptionId", "MisconceptionName"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(25)
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(
        top25["MisconceptionName"].str[:60],
        top25["count"],
        color="steelblue"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Training Samples")
    ax.set_title("Top 25 Misconceptions by Frequency", fontweight="bold")
    plt.tight_layout()
    save(fig, "03_top25_misconceptions.png")

    # ── 4. Subject distribution ────────────────────────────────────
    if "SubjectName" in train_df.columns:
        print("\n[5/7] Subject distribution...")
        subject_counts = train_df["SubjectName"].value_counts().head(20)
        print(f"  Unique subjects: {train_df['SubjectName'].nunique()}")
        print(f"  Top 3: {subject_counts.index[:3].tolist()}")

        fig, ax = plt.subplots(figsize=(10, 6))
        subject_counts.plot(kind="barh", ax=ax, color="coral")
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Top Subjects in Training Data", fontweight="bold")
        plt.tight_layout()
        save(fig, "04_subject_distribution.png")
    else:
        print("\n[5/7] SubjectName column not found — skipping subject plot.")

    # ── 5. Word cloud ──────────────────────────────────────────────
    print("\n[6/7] Word cloud...")
    try:
        from wordcloud import WordCloud, STOPWORDS
        text = " ".join(train_df["QuestionText"].dropna().tolist())
        stopwords = STOPWORDS | {"the", "a", "an", "of", "in", "is", "and", "to",
                                   "for", "with", "that", "are", "be", "as"}
        wc = WordCloud(
            width=1200, height=500, background_color="white",
            stopwords=stopwords, max_words=150,
            colormap="Blues"
        ).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud — Question Texts", fontweight="bold")
        save(fig, "05_wordcloud_questions.png")
    except ImportError:
        print("  wordcloud not installed — skipping.")

    # ── 6. Vocabulary overlap ──────────────────────────────────────
    print("\n[7/7] Vocabulary overlap analysis...")

    def tokenize(texts):
        words = set()
        for t in texts:
            if isinstance(t, str):
                words.update(re.sub(r"[^a-z0-9 ]", "", t.lower()).split())
        return words

    import re
    q_vocab = tokenize(train_df["QuestionText"].dropna())
    ans_vocab = tokenize(train_df["WrongAnswerText"].dropna())
    misc_vocab = tokenize(misconceptions["MisconceptionName"].dropna())

    overlap_q_misc = len(q_vocab & misc_vocab) / len(misc_vocab) if misc_vocab else 0
    overlap_ans_misc = len(ans_vocab & misc_vocab) / len(misc_vocab) if misc_vocab else 0

    print(f"  Question vocab size:          {len(q_vocab):,} words")
    print(f"  Wrong answer vocab size:      {len(ans_vocab):,} words")
    print(f"  Misconception vocab size:     {len(misc_vocab):,} words")
    print(f"  Overlap (Q ∩ Misc) / |Misc|:  {overlap_q_misc:.2%}")
    print(f"  Overlap (Ans ∩ Misc) / |Misc|:{overlap_ans_misc:.2%}")

    labels = ["Q ∩ Misc", "Ans ∩ Misc", "Q only", "Misc only"]
    vals = [
        len(q_vocab & misc_vocab),
        len(ans_vocab & misc_vocab),
        len(q_vocab - misc_vocab),
        len(misc_vocab - q_vocab),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color=["steelblue", "coral", "lightblue", "lightsalmon"])
    ax.set_ylabel("Vocabulary Size (words)")
    ax.set_title("Vocabulary Overlap: Questions & Answers vs. Misconceptions",
                 fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 20, f"{v:,}", ha="center", fontsize=9)
    save(fig, "06_vocab_overlap.png")

    # ── Summary text file ──────────────────────────────────────────
    summary_path = Path(__file__).parent / "eda" / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write("RAG-MAP EDA Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training pairs:              {n_train:,}\n")
        f.write(f"Unique questions:            {train_df['QuestionId'].nunique():,}\n")
        f.write(f"Total misconception classes: {n_misconceptions:,}\n")
        f.write(f"Misconceptions in train:     {train_df['MisconceptionId'].nunique():,}\n")
        f.write(f"Test rows:                   {len(test_df):,}\n\n")
        f.write(f"Label imbalance (Gini):      {gini:.4f}\n")
        f.write(f"Classes seen only once:      {(label_counts == 1).sum():,}\n\n")
        f.write(f"Question mean word count:    {train_df['q_len'].mean():.1f}\n")
        f.write(f"Misconception mean words:    {misconceptions['m_len'].mean():.1f}\n\n")
        f.write(f"Vocab overlap (Q∩Misc/Misc): {overlap_q_misc:.2%}\n")
        f.write("(Low overlap → lexical baselines will struggle)\n")
    print(f"\n  Summary written → {summary_path}")

    print("\n✓ EDA complete. Figures saved to eda/figures/")
    print("  → Use eda_summary.txt numbers in your checkpoint report.\n")


if __name__ == "__main__":
    run_eda()
