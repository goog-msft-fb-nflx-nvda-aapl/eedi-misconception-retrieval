"""
03_train_retriever.py
Fine-tune a bi-encoder on Eedi (query = question + wrong answer, doc = misconception name).

Loss: InfoNCE over [in-batch positives + 1 BM25 hard negative per anchor], temperature tau.
      In-batch columns whose MisconceptionId equals the anchor's are masked (false negatives).
Split: grouped by QuestionId (no question in both train and val).
Eval each epoch: MAP@3 / MAP@25 / Recall@25 on all val rows, plus the same on val rows whose
      misconception never occurs in train ("unseen") vs does ("seen").

Run:  python 03_train_retriever.py --model_name intfloat/multilingual-e5-small --tag e5s
"""
import sys, os, argparse, json, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils.helpers import (load_data, build_query, map_at_k, recall_at_k,
                           split_by_question, MODELS_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="intfloat/multilingual-e5-small")
    p.add_argument("--tag", default="e5s")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--hard_negatives", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expand", action="store_true", help="doc text = name + LLM expansion (06_expand_misconceptions.py)")
    p.add_argument("--q_prefix", default="query: ")
    p.add_argument("--d_prefix", default="passage: ")
    return p.parse_args()


def embed(model, texts, bs, grad=False):
    feats = model.preprocess(texts)
    feats = {k: (v.to(model.device) if hasattr(v, 'to') else v) for k, v in dict(feats).items()}
    out = model(feats)["sentence_embedding"]
    return F.normalize(out, dim=-1)


@torch.no_grad()
def encode_all(model, texts, bs=128):
    model.eval()
    outs = []
    for i in range(0, len(texts), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(embed(model, texts[i:i + bs], bs).float().cpu())
    return torch.cat(outs)


def evaluate(model, val_df, corpus, misc_ids, train_ids, args):
    q = encode_all(model, [args.q_prefix + x for x in val_df["query"]])
    d = encode_all(model, [args.d_prefix + x for x in corpus])
    sims = q @ d.T
    top = sims.topk(25, dim=1).indices.numpy()
    preds = [[misc_ids[j] for j in row] for row in top]
    labels = val_df["MisconceptionId"].tolist()
    res = {"MAP@3": map_at_k(preds, labels, 3), "MAP@25": map_at_k(preds, labels, 25),
           "Recall@25": recall_at_k(preds, labels, 25), "n": len(labels)}
    for name, keep in (("seen", lambda l: l in train_ids), ("unseen", lambda l: l not in train_ids)):
        idx = [i for i, l in enumerate(labels) if keep(l)]
        if idx:
            res[name] = {"n": len(idx),
                         "MAP@3": map_at_k([preds[i] for i in idx], [labels[i] for i in idx], 3),
                         "Recall@25": recall_at_k([preds[i] for i in idx], [labels[i] for i in idx], 25)}
    return res


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out_dir = MODELS_DIR / f"retriever_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, mis, _ = load_data()
    train_df["query"] = train_df.apply(build_query, axis=1)
    tr, va = split_by_question(train_df, a.val_size, a.seed)
    misc_ids = mis["MisconceptionId"].tolist()
    names = mis["MisconceptionName"].tolist()
    bm25_corpus = names  # BM25 hard negatives always mined on the short names
    if a.expand:
        import pandas as pd
        from utils.helpers import DATA_DIR
        ex = pd.read_csv(DATA_DIR / "misconception_expanded.csv").set_index("MisconceptionId")["Expansion"]
        corpus = [f"{n}. {ex[i]}" for n, i in zip(names, misc_ids)]
    else:
        corpus = names
    id2name = dict(zip(misc_ids, corpus))
    train_ids = set(tr["MisconceptionId"])
    print(f"train {len(tr)} val {len(va)} | val rows with unseen label: "
          f"{sum(l not in train_ids for l in va['MisconceptionId'])}", flush=True)

    # BM25 hard negatives (excluding the positive)
    hard = []
    if a.hard_negatives > 0:
        bm25 = BM25Okapi([c.lower().split() for c in bm25_corpus])
        for q, pid in tqdm(zip(tr["query"], tr["MisconceptionId"]), total=len(tr), desc="bm25 negs"):
            sc = bm25.get_scores(q.lower().split())
            cand = [misc_ids[i] for i in np.argsort(sc)[::-1][:50] if misc_ids[i] != pid]
            hard.append(cand[:a.hard_negatives])

    model = SentenceTransformer(a.model_name, device="cuda")
    model.max_seq_length = a.max_seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n = len(tr)
    steps = (n // a.batch_size) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))

    hist = [{"epoch": 0, **evaluate(model, va, corpus, misc_ids, train_ids, a)}]
    print("epoch 0 (zero-shot):", json.dumps(hist[-1]), flush=True)
    queries = tr["query"].tolist(); pids = tr["MisconceptionId"].tolist()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for b in range(n // a.batch_size):
            idx = perm[b * a.batch_size:(b + 1) * a.batch_size]
            qs = [a.q_prefix + queries[i] for i in idx]
            docs = [id2name[pids[i]] for i in idx]
            doc_ids = [pids[i] for i in idx]
            for i in idx:
                for h in (hard[i] if hard else []):
                    docs.append(id2name[h]); doc_ids.append(h)
            ds = [a.d_prefix + d for d in docs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qe = embed(model, qs, 0); de = embed(model, ds, 0)
            logits = (qe.float() @ de.float().T) / a.tau
            tgt = torch.arange(len(idx), device=logits.device)
            ids_t = torch.tensor(doc_ids, device=logits.device)
            anchor_ids = ids_t[:len(idx)]
            mask = (anchor_ids[:, None] == ids_t[None, :])
            mask[tgt, tgt] = False  # keep own positive
            logits = logits.masked_fill(mask, -1e4)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        r = evaluate(model, va, corpus, misc_ids, train_ids, a)
        hist.append({"epoch": ep, "loss": tot / (n // a.batch_size), **r})
        print(f"epoch {ep}: loss {hist[-1]['loss']:.4f} MAP@3 {r['MAP@3']:.4f} "
              f"MAP@25 {r['MAP@25']:.4f} R@25 {r['Recall@25']:.4f} "
              f"seen R@25 {r['seen']['Recall@25']:.3f} unseen R@25 {r.get('unseen',{}).get('Recall@25',float('nan')):.3f}",
              flush=True)

    model.save(str(out_dir))  # final-epoch weights (epochs fixed a priori, no val-based selection)
    json.dump({"args": vars(a), "history": hist}, open(out_dir / "history.json", "w"), indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
