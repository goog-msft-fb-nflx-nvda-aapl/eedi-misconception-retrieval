"""
04_train_reranker.py
Cross-encoder reranker over the fine-tuned retriever's top-25.

Training: per train query, group = [positive + (G-1) hard negatives sampled from retriever top-25],
          listwise softmax cross-entropy over the group (positive = target).
Eval:     val queries, candidates = retriever top-25; rerank; MAP@3 / MAP@25 (+ retriever-only reference).
Split:    identical grouped-by-question split as 03 (same seed).

Run:  python 04_train_reranker.py --retriever_tag e5s --tag mini
"""
import sys, os, argparse, json, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.helpers import load_data, build_query, map_at_k, recall_at_k, split_by_question, MODELS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    p.add_argument("--retriever_tag", default="e5s")
    p.add_argument("--tag", default="mini")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--queries_per_step", type=int, default=8)
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_seq_len", type=int, default=320)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=25)
    return p.parse_args()


@torch.no_grad()
def retrieve(retr, queries, corpus, k):
    def enc(texts):
        return retr.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
    d = enc(["passage: " + c for c in corpus]); q = enc(["query: " + x for x in queries])
    t = (q @ d.T).topk(k, dim=1)
    return t.indices.cpu().numpy(), t.values.float().cpu().numpy()


def score(model, tok, qs, ds, max_len, device):
    enc = tok(qs, ds, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**enc).logits.squeeze(-1).float()


@torch.no_grad()
def evaluate(model, tok, va, cand_idx, corpus, misc_ids, a, device):
    model.eval()
    labels = va["MisconceptionId"].tolist(); queries = va["query"].tolist()
    preds = []
    for q, cidx in zip(queries, cand_idx):
        s = score(model, tok, [q] * len(cidx), [corpus[j] for j in cidx], a.max_seq_len, device).cpu().numpy()
        preds.append([misc_ids[cidx[j]] for j in np.argsort(-s)])
    return {"MAP@3": map_at_k(preds, labels, 3), "MAP@25": map_at_k(preds, labels, 25),
            "Recall@25": recall_at_k(preds, labels, 25)}


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda"
    train_df, mis, _ = load_data()
    train_df["query"] = train_df.apply(build_query, axis=1)
    tr, va = split_by_question(train_df, a.val_size, a.seed)
    corpus = mis["MisconceptionName"].tolist(); misc_ids = mis["MisconceptionId"].tolist()
    id2idx = {m: i for i, m in enumerate(misc_ids)}

    retr = SentenceTransformer(str(MODELS_DIR / f"retriever_{a.retriever_tag}"), device=device)
    retr.max_seq_length = 192
    tr_cand, tr_sim = retrieve(retr, tr["query"].tolist(), corpus, a.topk)
    va_cand, va_sim = retrieve(retr, va["query"].tolist(), corpus, a.topk)
    ref = {"MAP@3": map_at_k([[misc_ids[j] for j in r] for r in va_cand], va["MisconceptionId"].tolist(), 3),
           "Recall@25": recall_at_k([[misc_ids[j] for j in r] for r in va_cand], va["MisconceptionId"].tolist(), 25)}
    print("retriever-only reference:", ref, flush=True)
    del retr; torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(a.reranker_model)
    model = AutoModelForSequenceClassification.from_pretrained(a.reranker_model, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n = len(tr); spe = n // a.queries_per_step; steps = spe * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))

    queries = tr["query"].tolist(); pids = tr["MisconceptionId"].tolist()
    hist = [{"epoch": 0, **evaluate(model, tok, va, va_cand, corpus, misc_ids, a, device)}]
    print("epoch 0 (untrained head):", hist[-1], flush=True)
    for ep in range(1, a.epochs + 1):
        model.train(); perm = np.random.permutation(n); tot = 0.0
        for b in range(spe):
            qs, ds, tgt_off = [], [], []
            for i in perm[b * a.queries_per_step:(b + 1) * a.queries_per_step]:
                pos = id2idx[pids[i]]
                negs = [j for j in tr_cand[i] if j != pos]
                negs = list(np.random.permutation(negs)[:a.group - 1])
                grp = [pos] + negs
                qs += [queries[i]] * len(grp); ds += [corpus[j] for j in grp]
            logits = score(model, tok, qs, ds, a.max_seq_len, device)
            # groups may be shorter than G only if <G-1 negatives (never with topk=25, G=16)
            logits = logits.view(a.queries_per_step, a.group)
            loss = F.cross_entropy(logits, torch.zeros(a.queries_per_step, dtype=torch.long, device=device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); tot += loss.item()
        r = evaluate(model, tok, va, va_cand, corpus, misc_ids, a, device)
        hist.append({"epoch": ep, "loss": tot / spe, **r})
        print(f"epoch {ep}: loss {tot/spe:.4f} MAP@3 {r['MAP@3']:.4f} MAP@25 {r['MAP@25']:.4f}", flush=True)

    # dump per-candidate scores on val for fusion / selector experiments
    model.eval(); rr = []
    with torch.no_grad():
        for q, cidx in zip(va["query"].tolist(), va_cand):
            rr.append(score(model, tok, [q] * len(cidx), [corpus[j] for j in cidx], a.max_seq_len, device).cpu().numpy())
    np.savez(MODELS_DIR / f"valscores_{a.tag}.npz", cand=va_cand, ret_sim=va_sim, rr=np.array(rr),
             cand_ids=np.array([[misc_ids[j] for j in r] for r in va_cand]), labels=va["MisconceptionId"].values,
             qid=va["QuestionId"].values)
    out = MODELS_DIR / f"reranker_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    json.dump({"args": vars(a), "retriever_only": ref, "history": hist}, open(out / "history.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
