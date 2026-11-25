"""
05_make_tables.py
Builds results/ablation.md from the saved histories in results/ (final-epoch rows).
No models are loaded; run it after 02/03/04.
"""
import json, glob, os, pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")
rows = []

for f in ("baseline_results.csv", "baseline_results_SubjectId.csv"):
    p = os.path.join(R, f)
    if os.path.exists(p):
        split = "SubjectId" if "SubjectId" in f else "QuestionId"
        for _, r in pd.read_csv(p).iterrows():
            rows.append((r["model"], split, r["MAP@3"], r["MAP@25"], r["Recall@25"]))

for f in sorted(glob.glob(os.path.join(R, "hist_ret_*.json"))):
    h = json.load(open(f)); tag = os.path.basename(f)[9:-5]
    split = "SubjectId" if "subj" in tag else "QuestionId"
    z, l = h["history"][0], h["history"][-1]
    rows.append((f"{tag} zero-shot (epoch 0)", split, z["MAP@3"], z["MAP@25"], z["Recall@25"]))
    rows.append((f"{tag} fine-tuned", split, l["MAP@3"], l["MAP@25"], l["Recall@25"]))

for f in sorted(glob.glob(os.path.join(R, "hist_rr_*.json"))):
    h = json.load(open(f)); tag = os.path.basename(f)[8:-5]
    split = "SubjectId" if "subj" in tag else "QuestionId"
    l = h["history"][-1]
    rows.append((f"{tag} reranker over its retriever", split, l["MAP@3"], l["MAP@25"], h["retriever_only"]["Recall@25"]))

df = pd.DataFrame(rows, columns=["config", "split", "MAP@3", "MAP@25", "Recall@25 (candidates)"])
df[["MAP@3", "MAP@25", "Recall@25 (candidates)"]] = df[["MAP@3", "MAP@25", "Recall@25 (candidates)"]].round(4)
out = os.path.join(R, "ablation.md")
open(out, "w").write(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))
print(open(out).read())
