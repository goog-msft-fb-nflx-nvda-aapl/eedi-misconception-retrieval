# Survey: prior work on Eedi "Mining Misconceptions in Mathematics" (Kaggle 2024)

Source type: competitor writeups (not peer reviewed), read 2026-11-25 via the nvidia-kaggle tool.
Task there: given MCQ + correct answer + one wrong answer, rank the top-25 of 2,587 misconceptions (MAP@25).
Base URL: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/

| Rank | Slug | Pipeline (as reported) |
|---|---|---|
| 1 | mth-101-1st-place-detailed-solution | retrieve top ~32-64, then fine-tuned Qwen2.5 14B -> 32B -> 72B rankers (top 8 -> 5); ~1.8k competition + ~10k synthetic examples; chain-of-thought distilled from a strong LLM; synthetic data stated as key for unseen misconceptions |
| 2 | cqyr-2nd-place-solution | 3 rounds of synthetic questions (LLM writes question, Qwen-math solves right/wrong answer, GPT-4o-mini quality filter); LLM-written explanation of each misconception (free at inference); Qwen2.5-32B chain-of-thought appended to the query |
| 3 | waseda-pochi-3rd-place-solution-with-magic-boost | Qwen-14B embedders -> top 35 -> Qwen-32B reranker, 6 LoRAs ensembled; post-processing that up-weights misconceptions unseen in train (reported public 0.590 -> 0.658, private 0.564 -> 0.600) |
| 4 | takoi-charmq-kami-4th-place-solution | synthetic data aimed at unseen misconceptions; LLM-generated misconception text added to input; LoRA Qwen 14B/32B retrievers; 32B rerankers; reserved retrieval slots for non-train misconceptions; seen-misconception scores x0.40; GroupKFold by QuestionId |
| 5 | ebi-ktr-5th-place-solution | synthetic data for misconceptions without problems; distilled reasoning from Qwen2.5-32B as input; listwise reranker (52 candidates per pass); GroupKFold by SubjectId matched the leaderboard better (reported local CV 0.626 vs LB 0.633) |

## Reported by multiple teams (their claims, not measured by us)
- Over 900 misconceptions in the taxonomy never occur in train; the hidden test was reported to be dominated by unseen misconceptions (3rd place probed roughly 3:1 unseen:seen).
- Synthetic data + LLM-written reasoning/explanations were the main levers; 14B-72B LLM rankers replaced small cross-encoders.
- Validation grouped by QuestionId overestimates the leaderboard; grouping by SubjectId is closer.

## Consequences for this project
- Our grouped-by-question split is the winners' weaker validation. Add a SubjectId-grouped split as the harder, more realistic number.
- Our scale (E5 + MiniLM/bge, one GPU) is a scaled-down version of this retrieve-then-rerank design; frame it as a controlled small-budget study, not a new method.
- Cheap winner ideas testable here: misconception expansion with LLM-written explanations; LLM reasoning appended to the query.

## Generic references already in the report
DPR (Karpukhin 2020), E5 (Wang 2024), cross-encoder reranking (Nogueira & Cho 2019), RAG (Lewis 2020), REPLUG (Shi 2024), FAISS (Johnson 2019).

## Cross-check against RESPONSE.md (external deep-research report, reference only)
Agrees with the writeups I read directly: retrieve-then-rerank with QLoRA Qwen 14B-72B; synthetic data; LLM reasoning in the query; offline LLM expansion of misconception text; unseen-label dominance in the test set; SubjectId grouping tracks LB better than QuestionId grouping.
Conflicts / unverified (do NOT cite as fact until checked against a primary source):
- RESPONSE.md says the 3rd-place team used no synthetic data; their own writeup says one model included ~2,000 GPT-generated samples (writeup wins).
- Rank 1 private LB 0.638, rank 1 GroupKFold key (ConstructId), efficiency-prize pipelines, and the 2025 arXiv 0.694 figure come only from secondary sources there.
- Refs 19-21 in RESPONSE.md (SkillFlow, Agentic AutoRAG, Medium serving post) are unrelated; not used.
Planned experiment order (one at a time): SubjectId-grouped split baseline -> offline misconception expansion -> query-side reasoning -> unseen-label calibration.
