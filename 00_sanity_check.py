"""
00_sanity_check.py
Run this FIRST after downloading the data to verify everything is in order.
Prints column names, shapes, and a few sample rows.

Run:  python 00_sanity_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def check():
    files = {
        "train.csv": None,
        "test.csv": None,
        "misconception_mapping.csv": None,
        "sample_submission.csv": None,
    }

    print("=" * 55)
    print("RAG-MAP: Data Sanity Check")
    print("=" * 55)

    all_ok = True
    for fname in files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  ✗ MISSING: {fpath}")
            all_ok = False
        else:
            df = pd.read_csv(fpath)
            files[fname] = df
            print(f"\n  ✓ {fname}")
            print(f"    Shape:   {df.shape}")
            print(f"    Columns: {df.columns.tolist()}")

    if not all_ok:
        print("\n⚠  Some files are missing. Run ./download_data.sh first.")
        return

    # Quick diagnostics
    train = files["train.csv"]
    misc  = files["misconception_mapping.csv"]

    print("\n── Train sample ──")
    print(train.head(3).to_string())

    print("\n── Misconception sample ──")
    print(misc.head(5).to_string())

    # Check for expected columns
    expected_train_cols = ["QuestionId", "QuestionText", "CorrectAnswer"]
    for col in expected_train_cols:
        if col not in train.columns:
            print(f"\n  ⚠  Column '{col}' not in train.csv — check column names!")

    # Check answer option columns
    answer_cols_found = [c for c in train.columns
                         if c.startswith("Answer") or c.startswith("Misconception")]
    print(f"\n  Answer/Misconception columns: {answer_cols_found}")

    print("\n\n→ If everything looks correct, proceed with:")
    print("   python 01_eda.py")
    print("   python 02_baselines.py")


if __name__ == "__main__":
    check()
