"""
utils/helpers.py
Shared utilities: data loading, MAP@K metric, text cleaning.
"""
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "eda" / "figures"

for d in [DATA_DIR, MODELS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────
def load_data():
    """
    Load and merge the competition CSVs into a single train DataFrame.

    Returns
    -------
    train_df : pd.DataFrame
        One row per (QuestionId_Answer) with columns:
        QuestionId, QuestionText, CorrectAnswer, AnswerText,
        MisconceptionId, MisconceptionName
    misconception_df : pd.DataFrame
        Full misconception mapping (2587 rows for Eedi dataset)
    test_df : pd.DataFrame
        Test set (no labels)
    """
    train = pd.read_csv(DATA_DIR / "train.csv")
    misconceptions = pd.read_csv(DATA_DIR / "misconception_mapping.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # The train CSV has columns for each answer option (A/B/C/D)
    # We need to "melt" to one row per distractor
    # Expected train columns: QuestionId, QuestionText,
    #   CorrectAnswer, Answer{A,B,C,D}Text, Misconception{A,B,C,D}Id
    # (adjust if column names differ)
    answer_cols = ["A", "B", "C", "D"]
    rows = []
    for _, row in train.iterrows():
        correct = row["CorrectAnswer"]
        for ans in answer_cols:
            if ans == correct:
                continue  # skip the correct answer
            misconception_id = row.get(f"Misconception{ans}Id", np.nan)
            if pd.isna(misconception_id):
                continue
            rows.append({
                "QuestionId": row["QuestionId"],
                "QuestionText": row["QuestionText"],
                "SubjectId": row.get("SubjectId", -1),
                "SubjectName": row.get("SubjectName", ""),
                "ConstructName": row.get("ConstructName", ""),
                "CorrectAnswer": correct,
                "CorrectAnswerText": row[f"Answer{correct}Text"],
                "WrongAnswer": ans,
                "WrongAnswerText": row[f"Answer{ans}Text"],
                "MisconceptionId": int(misconception_id),
            })
    train_df = pd.DataFrame(rows)

    # Merge misconception names
    train_df = train_df.merge(
        misconceptions.rename(columns={"MisconceptionId": "MisconceptionId",
                                       "MisconceptionName": "MisconceptionName"}),
        on="MisconceptionId", how="left"
    )

    return train_df, misconceptions, test


def build_query(row: pd.Series) -> str:
    """
    Construct the retrieval query string from a row.
    Combines: construct name + subject + question text + correct answer + wrong answer.
    This is the standard format used by top solutions.
    """
    parts = []
    if row.get("ConstructName"):
        parts.append(f"Construct: {row['ConstructName']}")
    if row.get("SubjectName"):
        parts.append(f"Subject: {row['SubjectName']}")
    parts.append(f"Question: {row['QuestionText']}")
    parts.append(f"Correct Answer: {row['CorrectAnswerText']}")
    parts.append(f"Wrong Answer: {row['WrongAnswerText']}")
    return " | ".join(parts)


def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── MAP@K metric ───────────────────────────────────────────────────────────
def average_precision_at_k(predicted: list, relevant: int, k: int = 3) -> float:
    """
    Compute Average Precision@K for a single query.

    Parameters
    ----------
    predicted : list of int
        Ranked list of predicted MisconceptionIds (best first).
    relevant : int
        The single correct MisconceptionId.
    k : int
        Cutoff rank (default 3 for MAP@3).

    Returns
    -------
    float : AP@K in [0, 1]
    """
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0
    for i, p in enumerate(predicted):
        if p == relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score  # at most 1 hit so this equals P@rank_of_hit / 1


def map_at_k(all_predicted: list, all_relevant: list, k: int = 3) -> float:
    """
    Compute MAP@K over the full validation set.

    Parameters
    ----------
    all_predicted : list of list of int
        Ranked predictions for each query.
    all_relevant : list of int
        Ground-truth MisconceptionId for each query.
    k : int
        Cutoff.

    Returns
    -------
    float : MAP@K
    """
    assert len(all_predicted) == len(all_relevant)
    scores = [
        average_precision_at_k(pred, rel, k)
        for pred, rel in zip(all_predicted, all_relevant)
    ]
    return float(np.mean(scores))


def recall_at_k(all_predicted: list, all_relevant: list, k: int = 25) -> float:
    """Recall@K — fraction of queries where the correct answer is in top-K."""
    hits = sum(
        1 for pred, rel in zip(all_predicted, all_relevant)
        if rel in pred[:k]
    )
    return hits / len(all_relevant)


# ── Grouped split (no question appears in both train and val) ──────────────
SPLIT_BY = os.environ.get("SPLIT_BY", "QuestionId")  # QuestionId (default) or SubjectId


def split_by_question(df, val_size=0.15, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df[SPLIT_BY]))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
