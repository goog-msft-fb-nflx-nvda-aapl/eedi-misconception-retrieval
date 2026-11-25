"""Proper label-distribution statistics (Gini coefficient over the full 2,587-class taxonomy)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from utils.helpers import load_data, split_by_question
df, mis, _ = load_data()
cnt = df["MisconceptionId"].value_counts().reindex(mis["MisconceptionId"]).fillna(0).values
def gini(x):
    x = np.sort(x.astype(float)); n = len(x); i = np.arange(1, n + 1)
    return float((2 * (i * x).sum()) / (n * x.sum()) - (n + 1) / n)
tr, va = split_by_question(df, 0.15, 42)
tr_ids = set(tr["MisconceptionId"])
s = {"rows": len(df), "questions": int(df.QuestionId.nunique()), "taxonomy": len(mis),
     "classes_in_train_csv": int((cnt > 0).sum()), "classes_never_in_train_csv": int((cnt == 0).sum()),
     "gini_coefficient_over_taxonomy": gini(cnt), "gini_over_observed_classes": gini(cnt[cnt > 0]),
     "max_count": int(cnt.max()), "median_count_observed": float(np.median(cnt[cnt > 0])),
     "seen_once": int((cnt == 1).sum()),
     "top10_share_of_rows": float(np.sort(cnt)[::-1][:10].sum() / cnt.sum()),
     "split_train_rows": len(tr), "split_val_rows": len(va),
     "val_rows_unseen_label": int(sum(l not in tr_ids for l in va.MisconceptionId)),
     "distractors_per_question_mean": float(len(df) / df.QuestionId.nunique())}
print(json.dumps(s, indent=1)); json.dump(s, open("results/label_stats.json", "w"), indent=1)
