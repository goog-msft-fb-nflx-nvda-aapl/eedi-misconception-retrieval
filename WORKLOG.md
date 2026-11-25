# RAG-MAP worklog (Eedi 2024 "Mining Misconceptions in Mathematics")

Server: gsm-gpu2, ~/dm/ragmap, env ~/envs/dm, GPU 0 (all 4 GPUs passed a tensor smoke test 2026-11-25).
Data: train.csv melted to 4,370 (question, wrong answer) rows, 1,869 questions, taxonomy 2,587 misconceptions.
Query = construct | subject | question | correct answer | wrong answer. Doc = misconception name (E5 prefixes query:/passage:).
All numbers: single seed (42), final-epoch weights (epoch count fixed a priori, no val-based checkpoint selection), single val split.
Val sizes are small (655 / 852 rows): treat differences below ~0.02 MAP@3 as not established.

## Split definitions
- QS = GroupShuffleSplit by QuestionId, 15% val: 3,715 train / 655 val; 170 val rows (26%) have a misconception unseen in train.
- SS = GroupShuffleSplit by SubjectId, 15% val: 3,518 train / 852 val; 491 val rows (58%) unseen label.
Rationale: top Eedi teams report test was dominated by unseen misconceptions and SubjectId-grouped CV tracked LB better (SURVEY.md; competitor claims).

## Label stats (results/label_stats.json)
Gini coefficient over 2,587-class taxonomy 0.666 (0.461 over the 1,604 observed classes); 983 classes never in train; 747 seen once; max count 54; median (observed) 2; top-10 labels = 7.6% of rows; 2.34 distractors/question.
(Earlier EDA "Gini 0.9983" was Gini impurity, mislabelled; not used.)

## Results, MAP@3 / Recall@25 (retrieval top-25), val

| Config | Split | MAP@3 | MAP@25 | R@25 |
|---|---|---|---|---|
| BM25 | QS | 0.0886 | 0.1134 | 0.371 |
| TF-IDF+cos | QS | 0.1099 | 0.1404 | 0.423 |
| E5-small zero-shot | QS | 0.0611 | 0.0867 | 0.301 |
| E5-small fine-tuned | QS | 0.2695 | 0.3250 | 0.763 |
| E5-small ft + MiniLM-L12 reranker | QS | 0.2913 | 0.3383 | (cands R@25 0.762) |
| E5-large zero-shot | QS | 0.1000 | 0.1287 | 0.385 |
| E5-large fine-tuned | QS | 0.3099 | 0.3636 | 0.809 |
| E5-large ft + MiniLM-L12 reranker | QS | 0.3079 | 0.3609 | (cands R@25 0.808) |
| E5-large ft + bge-reranker-v2-m3 | QS | 0.3394 | 0.3925 | (cands R@25 0.808) |
| BM25 | SS | 0.0869 | 0.1180 | 0.383 |
| TF-IDF+cos | SS | 0.1125 | 0.1516 | 0.466 |
| E5-small zero-shot | SS | 0.0826 | 0.1067 | 0.329 |
| E5-large zero-shot (names) | SS | 0.1164 | 0.1481 | 0.424 |
| E5-large zero-shot + LLM expansion | SS | 0.1371 | 0.1813 | 0.576 |
| E5-large fine-tuned (names) | SS | 0.2320 | 0.2816 | 0.714 |
| E5-large fine-tuned + LLM expansion | SS | 0.2248 | 0.2807 | 0.731 |
| E5-large ft (names) + bge-reranker-v2-m3 | SS | 0.2637 | 0.3091 | (cands R@25 0.710) |

Seen/unseen Recall@25, E5-large fine-tuned: QS seen 0.831 / unseen 0.747; SS names seen 0.668 / unseen 0.747; SS +expansion seen 0.698 / unseen 0.756.
The SS reranker's retriever-only reference is MAP@3 0.2320, R@25 0.710 (03 reported R@25 0.714: encoding precision). Retriever-only reference inside 04 differs from 03 in the 4th decimal (E5-small 0.2692 vs 0.2695, E5-large 0.3097 vs 0.3099): encoding precision, not a different model.

## Setup details
- Retriever (03): InfoNCE tau 0.05, in-batch + 1 BM25 hard negative per anchor, same-misconception in-batch false negatives masked, AdamW, bf16, batch 64, 10 epochs, lr 3e-5 (small) / 1.5e-5 (large), linear warmup/decay.
- Reranker (04): listwise softmax CE over [positive + 15 negatives from retriever top-25], 8 queries/step, 5 epochs; MiniLM lr 3e-5, bge lr 1e-5; val candidates = retriever top-25. Training candidates come from a retriever fine-tuned on those same rows (optimistic negatives).
- Expansion (06): Qwen2.5-7B-Instruct, greedy, <=90 new tokens, input = misconception name only; prompt in the script. Expansion run used max_seq_len 256 vs 192 for the names run (confound).
- Synthetic note: expansions are uneven in quality (e.g. id 0 explanation is muddled).

## Not done / open
- RL passage selector (proposal bonus): not run. As specified (always output 3 of a ranked list) it reduces to a ranking loss.
- Query-side LLM reasoning, unseen-label calibration, LLM (Qwen) rerankers, synthetic data: not run.
- Single seed; no confidence intervals.

## Logs / artifacts on server (~/dm/ragmap)
results/ (histories, label stats, expansion csv, valscores npz), models/ (retriever_*, reranker_*), *.log.
