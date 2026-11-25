"""
10_error_analysis.py
Rank-of-correct distribution and example queries for the final pipeline (subject split, reranker + retriever-score fusion w=0.5).
Examples are chosen by a fixed rule (first validation row, in file order, in each category), not by hand.
Run: SPLIT_BY=SubjectId python 10_error_analysis.py --scores models/valscores_e5l_subj_expqr_bge.npz --w 0.5
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from utils.helpers import load_data, split_by_question

p = argparse.ArgumentParser(); p.add_argument("--scores", required=True); p.add_argument("--w", type=float, default=0.5)
a = p.parse_args()
z = np.load(a.scores); cand, sim, rr, lab = z["cand_ids"], z["ret_sim"], z["rr"], z["labels"]
df, mis, _ = load_data(); tr, va = split_by_question(df, 0.15, 42)
train_ids = set(tr.MisconceptionId); name = dict(zip(mis.MisconceptionId, mis.MisconceptionName))
zs = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
s = zs(rr) + a.w * zs(sim); order = np.argsort(-s, 1); ranked = np.take_along_axis(cand, order, 1)
rank = np.array([int(np.where(ranked[i] == lab[i])[0][0]) + 1 if lab[i] in ranked[i] else 99 for i in range(len(lab))])
unseen = np.array([l not in train_ids for l in lab]); n = len(lab)
st = {"n": n, "top1": float((rank == 1).mean()), "top3": float((rank <= 3).mean()), "top25": float((rank <= 25).mean()),
      "top3_seen": float((rank[~unseen] <= 3).mean()), "top3_unseen": float((rank[unseen] <= 3).mean()),
      "n_seen": int((~unseen).sum()), "n_unseen": int(unseen.sum()), "rank2or3": float(((rank >= 2) & (rank <= 3)).mean()),
      "rank4to25": float(((rank >= 4) & (rank <= 25)).mean()), "not_in_top25": float((rank == 99).mean()),
      "in_sample_note": "fusion weight 0.5 fixed in advance from out-of-fold choice; this table is descriptive, not a new tuned score"}
print(json.dumps(st, indent=1)); json.dump(st, open("results/error_analysis.json", "w"), indent=1)
cats = {"correct at rank 1": lambda r: r == 1, "correct at rank 2 or 3": lambda r: (r >= 2) & (r <= 3), "correct in top 25 but below 3": lambda r: (r >= 4) & (r <= 25), "correct not in top 25": lambda r: r == 99}
out = ["# examples (fixed rule: first validation row in each category)\n"]
for c, f in cats.items():
    i = int(np.where(f(rank))[0][0]); r = va.iloc[i]
    out.append(f"## {c} (label {'unseen' if unseen[i] else 'seen'} in train, rank {rank[i] if rank[i] < 99 else '>25'})")
    out.append(f"- subject: {r.SubjectName}\n- question: {str(r.QuestionText)[:260].strip()}\n- correct answer: {r.CorrectAnswerText}\n- student wrong answer: {r.WrongAnswerText}")
    out.append(f"- true misconception: {name[lab[i]]}")
    for k in range(3): out.append(f"- predicted #{k+1}: {name[ranked[i][k]]}")
    out.append("")
open("results/error_examples.md", "w").write("\n".join(out)); print("\n".join(out))
