"""
09_rl_selector.py
RL passage selector (proposal bonus): a small policy scores each of the 25 reranked candidates from features
[z(reranker), z(retriever sim), rank features, unseen flag]; the final top-3 is sampled without replacement
(Plackett-Luce) and trained with REINFORCE, reward = AP@3, baseline = batch mean.
Compared on identical features with a supervised listwise policy (softmax cross-entropy) and the reranker alone.
2-fold CV over validation questions (train on one half, evaluate greedily on the other), 5 seeds.
Run: SPLIT_BY=SubjectId python 09_rl_selector.py --scores models/valscores_e5l_subj_bge.npz --tag subj
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from utils.helpers import load_data, split_by_question

p = argparse.ArgumentParser()
p.add_argument("--scores", required=True); p.add_argument("--tag", required=True)
p.add_argument("--steps", type=int, default=400); p.add_argument("--use_unseen", type=int, default=0)
a = p.parse_args()
torch.set_num_threads(2)
z = np.load(a.scores)
cand, sim, rr, lab, qid = z["cand_ids"], z["ret_sim"], z["rr"], z["labels"], z["qid"]
df, _, _ = load_data(); tr, _ = split_by_question(df, 0.15, 42); train_ids = set(tr.MisconceptionId)
unseen = np.vectorize(lambda m: m not in train_ids)(cand).astype(np.float32)
zs = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
rank = lambda x: np.argsort(np.argsort(-x, 1), 1) / 25.0
feats = [zs(rr), zs(sim), rank(rr), rank(sim)] + ([unseen] if a.use_unseen else [])
X = torch.tensor(np.stack(feats, -1), dtype=torch.float32)                    # (n, 25, f)
hit = torch.tensor((cand == lab[:, None]), dtype=torch.float32)             # (n, 25)
n = len(lab)

def ap3(order):  # order: (m, 25) candidate indices best-first
    top = order[:, :3]; h = torch.gather(hit[idx_eval], 1, top)
    return (h / torch.arange(1, 4)).sum(1)

def make(): return nn.Sequential(nn.Linear(X.shape[-1], 16), nn.Tanh(), nn.Linear(16, 1))

def train_eval(tr_idx, te_idx, mode, seed):
    global idx_eval
    torch.manual_seed(seed); np.random.seed(seed); net = make(); opt = torch.optim.Adam(net.parameters(), lr=3e-2)
    for step in range(a.steps):
        b = torch.tensor(np.random.choice(tr_idx, 128)); s = net(X[b]).squeeze(-1)
        if mode == "sup":
            has = hit[b].sum(1) > 0
            if has.any():
                loss = F.cross_entropy(s[has], hit[b][has].argmax(1))
            else: continue
        else:
            g = -torch.log(-torch.log(torch.rand_like(s)))                 # Gumbel top-k == Plackett-Luce sampling
            order = torch.argsort(s.detach() + g, 1, descending=True)[:, :3]
            logp = 0.0; sc = s; mask = torch.zeros_like(s, dtype=torch.bool)
            for k in range(3):
                lp = F.log_softmax(sc.masked_fill(mask, -1e9), 1); pick = order[:, k]
                logp = logp + lp.gather(1, pick[:, None]).squeeze(1); mask = mask.scatter(1, pick[:, None], True)
            r = (torch.gather(hit[b], 1, order) / torch.arange(1, 4)).sum(1)
            loss = -((r - r.mean()) * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        idx_eval = torch.tensor(te_idx); order = torch.argsort(net(X[idx_eval]).squeeze(-1), 1, descending=True)
        return ap3(order).mean().item()

uq = np.unique(qid); rng = np.random.RandomState(0); rng.shuffle(uq); half = set(uq[: len(uq) // 2])
fold = np.array([q in half for q in qid]); folds = [np.where(fold)[0], np.where(~fold)[0]]
out = {}
for mode in ("reranker_only", "supervised_listwise", "reinforce"):
    vals = []
    for seed in range(5):
        sc = []
        for k in (0, 1):
            te, trn = folds[k], folds[1 - k]
            if mode == "reranker_only":
                idx_eval = torch.tensor(te); sc.append((ap3(torch.argsort(torch.tensor(rr[te]), 1, descending=True)).mean().item(), len(te)))
            else:
                sc.append((train_eval(trn, te, "sup" if mode == "supervised_listwise" else "rl", seed), len(te)))
        vals.append(sum(s * m for s, m in sc) / sum(m for _, m in sc))
    out[mode] = {"MAP@3_oof_mean": float(np.mean(vals)), "std_over_seeds": float(np.std(vals)), "use_unseen": a.use_unseen}
    print(mode, json.dumps(out[mode]), flush=True)
json.dump(out, open(f"results/rl_{a.tag}_unseen{a.use_unseen}.json", "w"), indent=1)
