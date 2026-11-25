"""
08_calibration.py
Post-hoc score fusion + unseen-label bonus on the saved top-25 candidate scores (reranker + retriever).
score = z(reranker) + w * z(retriever sim) + b * [candidate label unseen in train]
(w, b) are chosen on one half of the validation questions and evaluated on the other half (2-fold, swapped);
only out-of-fold numbers are reported as results. In-sample best is printed for reference only.
Run: SPLIT_BY=SubjectId python 08_calibration.py --scores models/valscores_e5l_subj_bge.npz --tag subj
"""
import sys, os, argparse, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from utils.helpers import load_data, split_by_question, map_at_k

p = argparse.ArgumentParser()
p.add_argument("--scores", required=True); p.add_argument("--tag", required=True)
a = p.parse_args()
z = np.load(a.scores)
cand, sim, rr, lab, qid = z["cand_ids"], z["ret_sim"], z["rr"], z["labels"], z["qid"]
df, _, _ = load_data(); tr, va = split_by_question(df, 0.15, 42)
train_ids = set(tr.MisconceptionId)
unseen = np.vectorize(lambda m: m not in train_ids)(cand).astype(float)
zs = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
Zr, Zs = zs(rr), zs(sim)

def evaluate(idx, w, b):
    s = Zr[idx] + w * Zs[idx] + b * unseen[idx]
    order = np.argsort(-s, axis=1)
    preds = [list(cand[idx][i][order[i]]) for i in range(len(idx))]
    return map_at_k(preds, list(lab[idx]), 3), float(np.mean([unseen[idx][i][order[i][0]] for i in range(len(idx))]))

W = [0, 0.25, 0.5, 1.0]; B = [0, 0.25, 0.5, 1.0, 1.5, 2.0]
rng = np.random.RandomState(0)
uq = np.unique(qid); rng.shuffle(uq); half = set(uq[: len(uq) // 2])
fold = np.array([q in half for q in qid]); folds = [np.where(fold)[0], np.where(~fold)[0]]
res = {}
for name, grid in (("reranker only", [(0, 0)]), ("+ retriever fusion", [(w, 0) for w in W]), ("+ fusion + unseen bonus", list(itertools.product(W, B)))):
    oof = []; chosen = []
    for k in (0, 1):
        tune, test = folds[1 - k], folds[k]
        best = max(grid, key=lambda g: evaluate(tune, *g)[0])
        oof.append((evaluate(test, *best)[0], len(test), evaluate(test, *best)[1])); chosen.append(best)
    n = sum(o[1] for o in oof)
    res[name] = {"MAP@3_oof": sum(o[0] * o[1] for o in oof) / n, "unseen_in_top1": sum(o[2] * o[1] for o in oof) / n, "chosen_(w,b)": chosen}
    print(name, json.dumps(res[name]))
allidx = np.arange(len(lab))
best = max(itertools.product(W, B), key=lambda g: evaluate(allidx, *g)[0])
res["in_sample_best_reference_only"] = {"(w,b)": best, "MAP@3": evaluate(allidx, *best)[0]}
res["val_rows_label_unseen_share"] = float(np.mean([m not in train_ids for m in lab]))
res["val_rows_label_in_top25"] = float(np.mean([lab[i] in cand[i] for i in range(len(lab))]))
print(json.dumps({k: res[k] for k in list(res)[3:]}))
json.dump(res, open(f"results/calibration_{a.tag}.json", "w"), indent=1)
