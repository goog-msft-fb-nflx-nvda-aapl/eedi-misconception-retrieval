# Retrieving the misconception behind a wrong answer: a small-budget retrieve-and-rerank study on the Eedi data

James Christian. Data Mining final project, University of Colorado Boulder.
Code, logs and all numbers: https://github.com/goog-msft-fb-nflx-nvda-aapl/eedi-misconception-retrieval

## Abstract

Given a multiple-choice maths question, the correct answer and the wrong answer a student chose, the task is to rank the misconception behind the wrong answer out of a taxonomy of 2,587. I study a retrieve-then-rerank pipeline (fine-tuned E5 bi-encoder, BGE cross-encoder reranker) on the Kaggle Eedi 2024 data, with two offline language-model enrichments (an explanation for each misconception, a short reasoning for each wrong answer). On a subject-grouped validation split, where 58% of validation rows carry a misconception unseen in training, MAP@3 rises from 0.087 (BM25) and 0.118 (zero-shot E5-large) to 0.311 for the full pipeline with out-of-fold score fusion. Grouping by question instead of subject overstates the result by about eight points. A reinforcement-learning selector and an unseen-label bonus add nothing beyond simple score blending. Results are single-seed on 655 and 852 validation rows, so only large gaps are claimed. This is a scaled-down controlled study, not a leaderboard entry and not a new method.

## 1. Problem

Students often hold systematic wrong ideas, for example believing 0.355 > 0.8 because 355 > 8. Diagnosing the misconception from a wrong answer is slow for teachers. I treat it as ranked retrieval: the query is the question context plus the wrong answer, the documents are the misconception names, and success means the right misconception is in the top three. Retrieval rather than classification matters because the label space is large and many labels are rare or absent in training.

## 2. Data and exploratory analysis

Source: Kaggle "Eedi: Mining Misconceptions in Mathematics" (2024). The train file has 1,869 questions, each with up to three labelled wrong answers; reshaped, that is 4,370 (question, wrong answer) queries. The taxonomy has 2,587 misconceptions. The official test file is a three-row stub, so all evaluation uses held-out training data.

Label statistics (results/label_stats.json): 983 of 2,587 misconceptions (38%) never occur as a training label, 747 occur once, 857 occur twice or more. The Gini coefficient of label counts over the taxonomy is 0.67; the most frequent label has 54 rows and the ten most frequent cover 7.6% of rows. Questions average about 29 words and misconception names about 12. A classifier over seen labels cannot output 38% of the taxonomy, which motivates retrieval.

The query string is: construct | subject | question | correct answer | wrong answer. E5 query/passage prefixes are used.

## 3. Methods

Retriever (03_train_retriever.py): multilingual-e5-small or -large, InfoNCE with temperature 0.05, in-batch negatives plus one BM25 hard negative per query, same-misconception in-batch pairs masked as false negatives, AdamW, bf16, batch 64, 10 epochs, last-epoch weights (no validation-based checkpoint selection).

Reranker (04_train_reranker.py): cross-encoder over the retriever's top 25, listwise softmax cross-entropy over one positive and 15 retrieved negatives, 5 epochs. Backbones: ms-marco-MiniLM-L-12-v2 and BAAI/bge-reranker-v2-m3.

Offline enrichment (06, 07): Qwen2.5-7B-Instruct, greedy decoding. Each misconception name is expanded to a two-sentence explanation (input: the name only). Each wrong answer gets a short reasoning about the likely mistake (input: question, correct answer, wrong answer; no label). The explanation is appended to the document text, the reasoning to the query.

Post-hoc (08, 09): retriever and reranker scores are z-normalised per query and blended with weight w; an optional bonus is added for candidates whose label is unseen in training. Settings are tuned on one random half of validation questions and scored on the other half (two folds, swapped); only these out-of-fold numbers are reported. A REINFORCE selector (Plackett-Luce sampling of a top-3, reward AP@3) is compared with a supervised listwise policy on identical features.

## 4. Evaluation protocol

Metrics: MAP@3 (one correct label per query, so credit is 1, 1/2, 1/3 at ranks 1 to 3), plus MAP@25 and Recall@25 (the ceiling for the reranker). Two grouped 15% validation splits, seed 42: by question (655 validation rows, 26% with an unseen label) and by subject (852 rows, 58% unseen). The subject split is treated as the realistic one, following competitor reports that the hidden Eedi test set was dominated by unseen misconceptions and that subject-grouped validation tracked the leaderboard better. Everything is single-seed. Re-running the same configuration at a different input length moved MAP@3 by about 0.015, so differences below roughly 0.02 are treated as not established.

## 5. Results

Question split (MAP@3): BM25 0.089, TF-IDF 0.110, E5-small zero-shot 0.061, E5-large zero-shot 0.100, E5-small fine-tuned 0.270, E5-large fine-tuned 0.310 (Recall@25 0.81). Reranking: MiniLM lifts E5-small to 0.291 but leaves E5-large at 0.308; BGE-reranker-v2-m3 lifts E5-large to 0.339.

Subject split, same E5-large model fine-tuned on names only: 0.232, against 0.310 on the question split. On the subject split Recall@25 is 0.75 for rows with an unseen label and 0.67 for seen ones; the unseen rows are different questions, so the direction of that gap should not be over-read.

Enrichment ablation, subject split, E5-large, all at 320 tokens: names only 0.217 (Recall@25 0.710); with misconception explanations 0.228 (0.739); with answer reasoning 0.222 (0.755); with both 0.255 (0.782). Single additions are within the measured noise on MAP@3; the combination (+0.038 MAP@3, +0.07 Recall@25) is larger than it. Zero-shot, explanations alone raise Recall@25 from 0.42 to 0.58.

Final ladder, subject split (MAP@3): BM25 0.087, TF-IDF 0.112, E5-large zero-shot 0.118, fine-tuned 0.217, with both enrichments 0.255, plus BGE reranker 0.292, plus out-of-fold score fusion 0.311. The reranker's validation score varied by about 0.02 between epochs (peak at epoch 2, final epoch reported), so its roughly 0.03 gain is real in direction but imprecise.

Where the correct misconception lands (final pipeline, 852 queries): rank 1 24%, rank 2 or 3 18% (so 42% in the top three), rank 4 to 25 37%, not in the top 25 22%. Top-3 accuracy is 45% for seen and 39% for unseen labels. Four examples chosen by a fixed rule are in results/error_examples.md.

Negative results (out-of-fold, final scores): the unseen-label bonus gives 0.3085 vs 0.3110 without it; the REINFORCE selector gives 0.312, the supervised listwise policy 0.317, plain fusion 0.311, so the RL machinery adds nothing beyond blending two scores. The small MiniLM reranker does not help the large retriever. Misconception explanations alone do not clearly help after fine-tuning.

## 6. Discussion

Zero-shot embedding models are no better than keyword search here, and fine-tuning on about 3,700 pairs roughly triples MAP@3, so the domain gap is large but learnable. Validation design changes the headline by eight points, which is why the competition teams argued about it. Offline language-model enrichment helps recall most, plausibly because wrong-answer text and misconception names share little vocabulary; I did not test that explanation directly. A reranker needs enough capacity to help, which matches the competition winners' move to large language-model rankers, though I did not test the capacity explanation directly either. With exactly one right label per query, a learned selector has little to add beyond a well-blended score.

## 7. Related work

Dense passage retrieval (Karpukhin et al., 2020) and E5 (Wang et al., 2024) provide the bi-encoder; cross-encoder reranking (Nogueira and Cho, 2019; Chen et al., 2024 for BGE) provides the second stage; retrieval-augmented generation (Lewis et al., 2020) popularised retrieve-then-use. The top five public Eedi writeups (competitor reports, not peer reviewed; SURVEY.md) used the same two-stage shape with Qwen2.5 14B to 72B retrievers and rerankers, around ten thousand synthetic questions, large-model reasoning in the query, and LLM-written misconception explanations. This study reproduces the shape at 0.03 to 0.6B parameters on one GPU as a controlled ablation.

## 8. Limitations

Single seed, no confidence intervals; 655 and 852 validation rows; the validation split is also the reporting split (epochs fixed in advance, no checkpoint selection); generated explanations and reasoning from a 7B model are unaudited; reranker training negatives come from a retriever trained on the same rows; scores are not comparable to the Kaggle leaderboard (hidden, mostly unseen test set, much larger models).

## 9. Future work

Language-model rerankers (for example Qwen 14B with LoRA); synthetic questions for the 983 untrained misconceptions; more seeds and cross-validated subject splits; an audit of the generated text against held-out labelled examples; a teacher-in-the-loop workflow in which confirmed labels grow the training set.

## References

- Karpukhin et al. 2020. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- Wang et al. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
- Nogueira and Cho. 2019. Passage Re-ranking with BERT. arXiv:1901.04085.
- Chen et al. 2024. BGE M3-Embedding. arXiv:2402.03216.
- Lewis et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Qwen Team. 2024. Qwen2.5 Technical Report. arXiv:2412.15115.
- Eedi. 2024. Mining Misconceptions in Mathematics, Kaggle competition, and its top-5 public writeups.
