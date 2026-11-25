# presentation brief: eedi misconception retrieval

input for building the slides (canva) and for recording the video.
course ask: 20 to 30 minute recorded talk, good overview, key findings, future work, clear communication.
the script below is about 3,000 words, roughly 22 minutes at a normal speaking pace.
every number comes from `WORKLOG.md` / `results/` in this repo (single seed, 655 or 852 validation rows).

## design brief for the slide builder
- 16:9, 17 narrated slides plus one references slide, speaker notes = the script under each slide.
- topic is school maths: chalkboard feel. suggested palette: deep green (1F4D45) and charcoal (1C2B2A) for dark slides, white and pale grey-green (EEF3F1) for content slides, one amber accent (F2A33A), teal (2A9D8F) for secondary series.
- fonts: a serif for titles (cambria or similar), a clean sans for body (calibri or similar). titles 32pt or larger, body 16pt or larger.
- one idea per slide, a visual on every slide (chart, diagram, stat callout). no text walls: the on-slide text below is already the short version.
- charts must be real charts with the exact values given. keep chart styling identical across slides.
- do not add accent lines under titles or colored edge stripes.

---

## slide 1. title (dark)
on slide:
- title: Retrieving the Misconception Behind a Wrong Answer
- subtitle: a small-budget retrieve-and-rerank study on the Eedi data
- James Christian, Data Mining, University of Colorado Boulder
visual: a wrong-answer example as a faint background motif, for instance "0.355 > 0.8 ?"

script:
Hello, I'm James Christian, and this is my final project for Data Mining. The title is "Retrieving the Misconception Behind a Wrong Answer", a small-budget retrieve-and-rerank study on the Eedi data. In the next twenty or so minutes I will explain the problem, show the data and what it looks like, describe the pipeline I built, and then walk through the results, including the things that did not work. One thing I want to be upfront about: the task and the data come from the Kaggle Eedi competition from 2024. I am not claiming a new method, and I am not comparing myself to the leaderboard. What I can offer is a controlled, reproducible look at what a modest budget buys you on this problem, with every number measured by me and all the code and logs in a public repository.

## slide 2. the problem
on slide:
- title: Why a wrong answer needs a diagnosis
- flow of three boxes: wrong answer "0.355 > 0.8" -> misconception "longer number is bigger" -> targeted feedback
- one line: given the question, the correct answer and the wrong answer, rank the likely misconception. goal: right one in the top 3.
visual: three-step flow with the decimal example

script:
Let me start with the problem. Students aged nine to sixteen often develop systematic wrong ways of thinking, called misconceptions. A classic example: a student decides that 0.355 is bigger than 0.8, because 355 is bigger than 8. They are applying whole-number intuition to decimals. If a teacher knows the student has that misconception, the feedback can be targeted: here is why decimals do not work like that. The trouble is that working out the misconception from a wrong answer is slow, and it does not scale across a classroom of thirty students and many questions. So the data mining question is this. Given a multiple-choice maths question, the correct answer, and the wrong answer a student picked, can we automatically rank which misconception most likely explains that wrong answer? I frame it as ranked retrieval rather than flat classification, because there are over two and a half thousand possible misconceptions, and many of them are rare or never appear in the training data. The goal is that the right misconception shows up in the top three.

## slide 3. the data
on slide:
- title: The Eedi data
- stat callouts: 1,869 questions | 4,370 wrong-answer queries | 2,587 misconceptions in the taxonomy
- example card. query: construct, subject, question, correct answer, wrong answer. targets from the data: "Uses dividing fractions method for multiplying fractions", "Believes there are 100 degrees in a full turn"
- footnote: source, Kaggle Eedi "Mining Misconceptions in Mathematics" (2024). the official test file is a 3-row stub, so evaluation uses held-out training data.

script:
Here is the data. It comes from the Eedi competition on Kaggle. Each row is a question with four options, one correct and up to three wrong. For every wrong option an expert has labelled the misconception behind it, chosen from a fixed taxonomy of 2,587 misconceptions. After reshaping the table into one row per wrong answer I have 4,370 queries from 1,869 questions, about 2.3 wrong answers per question. The query I give the model is the construct and subject names, the question text, the correct answer and the wrong answer, joined into one string. The target is a short sentence, for example "Uses dividing fractions method for multiplying fractions", or "Believes there are 100 degrees in a full turn". The official test file in this competition is only a three-row stub, so all my evaluation is on held-out parts of the training data. I will come back to how I split it, because that turns out to matter a lot.

## slide 4. what the labels look like
on slide:
- title: A long tail, and 38% of labels never trained on
- bar chart, misconceptions by number of training rows. categories and values: "0 rows" 983, "1 row" 747, "2 or more rows" 857
- side facts: Gini coefficient of label counts over the taxonomy 0.67; most common label 54 rows; top 10 labels under 8% of rows; questions average 29 words, misconception names 12
visual: native bar chart plus 3 to 4 short facts

script:
Before modelling I looked at the labels, because they decide what is achievable. Of the 2,587 misconceptions in the taxonomy, 983 never occur as a label in the training file at all. Another 747 occur exactly once, and only 857 occur two or more times. The Gini coefficient of the label counts over the whole taxonomy is 0.67, so the distribution is heavily long-tailed, although the biggest single label is only 54 rows and the ten most common together are under 8 percent of the data. Two consequences. First, a classifier over seen labels can never output 38 percent of the taxonomy. That is why I use retrieval: the misconception text is an input, so in principle an unseen misconception can still be matched. Second, how I split validation decides whether I measure that ability or hide it. The texts themselves are short: questions average about 29 words and misconception names about 12.

## slide 5. related work
on slide:
- title: Where this sits
- three-row table (source | what they did | how this project differs):
  - retrieval methods: DPR, E5 bi-encoders; cross-encoder reranking (Nogueira and Cho; BGE); retrieve-then-use from RAG | used as building blocks, fine-tuned on this task
  - Eedi top-5 writeups (Kaggle, competitor reports): Qwen2.5 14B to 72B retrievers and rerankers, ~10k synthetic questions, LLM reasoning in the query, LLM-written misconception explanations | scaled down: models of 0.03 to 0.6B parameters, one GPU, controlled ablation
  - what they reported about validation: test set mostly unseen misconceptions; subject-grouped validation tracked the leaderboard better | I use both a question split and a subject split

script:
Related work comes from three places. Methodologically, dense passage retrieval showed that you can encode queries and passages independently with a bi-encoder and search by similarity. E5 gives strong general-purpose embeddings. Cross-encoder rerankers, such as Nogueira and Cho's BERT reranking and the BGE reranker, score a query and a candidate jointly, which is slower but more precise. Retrieval-augmented generation popularised the retrieve-then-use pattern. Second, the Eedi competition itself. I read the top five public writeups. The winners used the same two-stage shape but with much bigger models: Qwen 2.5 retrievers and rerankers from 14 to 72 billion parameters, around ten thousand synthetic training questions, and large-model reasoning added to the query. Two teams also had language models write explanations for each misconception. Several reported that the hidden test set was dominated by misconceptions unseen in training, and one said that validating by subject tracked the leaderboard better than validating by question. Those are competitor reports, not peer-reviewed results. Third, my contribution relative to that is a scaled-down version: models between about 0.03 and 0.6 billion parameters on one GPU, run as a controlled ablation instead of a leaderboard attempt.

## slide 6. the pipeline
on slide:
- title: Retrieve, then rerank
- diagram: query -> E5 bi-encoder (misconceptions embedded once) -> top 25 -> BGE cross-encoder -> top 3
- two side boxes feeding the diagram, "offline, no labels used": (1) 2-sentence explanation of each misconception, (2) short reasoning for each wrong answer. both from Qwen2.5-7B-Instruct.
- training note: contrastive loss with in-batch + 1 BM25 hard negative, same-misconception negatives masked; listwise softmax loss for the reranker

script:
Here is the pipeline. A query goes into a bi-encoder, multilingual E5, which embeds it. The 2,587 misconceptions are embedded once. I take the top 25 by cosine similarity. A cross-encoder reranker then reads the query together with each of the 25 candidates and re-scores them, and the top three are my answer. For training, the bi-encoder uses a contrastive loss with in-batch negatives plus one BM25 hard negative per query, and I mask out cases where two items in a batch share the same misconception, since those are not true negatives. The reranker uses a listwise softmax loss over one positive and fifteen retrieved negatives. On top of that, two offline enrichments using Qwen2.5 7B Instruct. One: expand each misconception with a two-sentence explanation, which costs nothing at inference time. Two: generate a short reasoning for each wrong answer, from the question, correct answer and wrong answer only. Neither uses the labels. I also test a reinforcement-learning selector, which I will come to.

## slide 7. how I evaluate
on slide:
- title: Metrics and two validation splits
- left: MAP@3 (one right answer: rank 1 = 1, rank 2 = 1/2, rank 3 = 1/3, else 0). also MAP@25 and Recall@25 (the reranker's ceiling)
- right: two stat cards. question split: 26% of validation rows have a label unseen in training (170 of 655). subject split: 58% (491 of 852)
- footer: single seed, fixed epochs, last epoch reported, no checkpoint picking. differences under ~0.02 MAP@3 are not established.

script:
How I evaluate. The main metric is mean average precision at three. Because each query has exactly one right misconception, you get full credit for rank one, a half for rank two, a third for rank three, and nothing otherwise. I also report MAP at 25 and recall at 25, because recall at 25 is the ceiling the reranker works under. I use two validation splits, each 15 percent, grouped so related rows stay together. The question split keeps all rows of one question on one side. The subject split holds out whole curriculum topics. The difference is large. On the question split, 26 percent of validation rows carry a misconception that never appears in training. On the subject split it is 58 percent. I treat the subject split as the realistic one. Everything is a single seed with 655 or 852 validation rows, and I train for a fixed number of epochs and report the last one, without picking the best checkpoint on validation. So I treat differences below about 0.02 MAP at 3 as not established.

## slide 8. baselines and fine-tuning (question split)
on slide:
- title: Zero-shot fails, fine-tuning triples it
- horizontal bar chart, MAP@3, question split: BM25 0.089 | TF-IDF 0.110 | E5-small zero-shot 0.061 | E5-large zero-shot 0.100 | E5-small fine-tuned 0.270 | E5-large fine-tuned 0.310
- callout: E5-large fine-tuned Recall@25 = 0.81

script:
First results, on the question split. The lexical baselines are weak: BM25 gets 0.089 MAP at 3, TF-IDF 0.110. Zero-shot E5, without any training, is no better: 0.061 for the small model and 0.100 for the large one. That matches what the competition organisers said, that pre-trained models did not work out of the box. Fine-tuning changes the picture. Small E5 goes to 0.270 and large E5 to 0.310. That is roughly a threefold improvement over zero-shot, and recall at 25 for the large model is 0.81, meaning the right misconception is in the candidate list for four of five queries. So the domain gap is real but learnable: with only about 3,700 training pairs, the encoder learns what the similarity between a wrong answer and its misconception looks like.

## slide 9. reranking (question split)
on slide:
- title: A reranker helps only if it is big enough
- column chart, MAP@3, question split: E5-small fine-tuned 0.270 | + MiniLM reranker 0.291 | E5-large fine-tuned 0.310 | + MiniLM reranker 0.308 | + BGE-reranker-v2-m3 0.339
- takeaway line: small reranker: +0.02 on the small retriever, nothing on the large one. BGE (~570M parameters): +0.03.

script:
Next, the reranker, on the same question split. With the small retriever, adding a MiniLM cross-encoder improves MAP at 3 from 0.270 to 0.291. With the large retriever, the same MiniLM reranker does not help: 0.308 versus 0.310. So a small general reranker adds nothing once the retriever is stronger. Switching to a larger multilingual reranker, BGE reranker v2 m3 at around 570 million parameters, gives 0.339, a gain of about 0.03 over the retriever alone. My reading, and I label it as interpretation, is that the reranker needs enough capacity to separate near-duplicate misconceptions, which is what the competition winners described when they moved to big language-model rankers. I have not tested that explanation directly.

## slide 10. the realistic split
on slide:
- title: The same model scores 8 points lower on new topics
- column chart, E5-large fine-tuned (names only), MAP@3: question split 0.310 | subject split 0.232
- callout: on the subject split, Recall@25 on unseen labels 0.75 (seen labels 0.67)

script:
Now the realistic split. Same model, same training recipe: the large E5 retriever fine-tuned on misconception names only. It scores 0.310 MAP at 3 on the question split and 0.232 on the subject split. It loses eight points, because on the question split it has seen other questions on the same topics during training. This is why the competition teams argued about validation. If I had only reported the question split, I would be overstating how well this works on new topics. One encouraging detail: on the subject split, recall at 25 for rows with unseen misconceptions is 0.75, not lower than the 0.67 for seen ones, so the retrieval formulation does reach labels the model never trained on. The unseen rows are different questions, so I would not read too much into the direction of that gap. I use the subject split for everything that follows.

## slide 11. LLM enrichment (subject split)
on slide:
- title: Both enrichments together: +0.038 MAP@3
- two side-by-side column charts, subject split, E5-large fine-tuned, same input length. categories: names only | + misconception explanations | + answer reasoning | + both
  - MAP@3: 0.217 | 0.229 | 0.222 | 0.255
  - Recall@25: 0.710 | 0.739 | 0.755 | 0.782
- callout: zero-shot Recall@25 with explanations 0.42 -> 0.58, with no training
- caveat line: run-to-run gap from input length alone was 0.015

script:
Now the language-model enrichments, all on the subject split and at the same input length, so the comparison is clean. Names only: 0.217 MAP at 3 and a recall at 25 of 0.71. Adding the explanations to the misconceptions: 0.229 and 0.74. Adding the generated reasoning to the query: 0.222 and 0.75. Both together: 0.255 and 0.78. Each addition alone is within the noise I measured, about 0.015 just from changing the input length, but both lift recall, and together the gain of 0.038 MAP at 3 and seven points of recall is larger than that noise. Zero-shot, the explanations matter most: recall at 25 jumps from 0.42 to 0.58 with no training at all. That fits what the top Eedi teams reported, here at the scale of a seven-billion-parameter generator. One caveat: the generated text is uneven. Some explanations are muddled, and I have not measured their quality.

## slide 12. the full ladder (subject split)
on slide:
- title: From keyword search to 0.31
- horizontal bar chart, MAP@3, subject split, ordered: BM25 0.087 | TF-IDF 0.112 | E5-large zero-shot 0.118 | fine-tuned 0.217 | + both enrichments 0.255 | + BGE reranker 0.292 | + score fusion 0.311
- footnote: fusion weight chosen on one half of validation questions and scored on the other (out-of-fold). 78% of queries have the right misconception in the top 25.

script:
Putting it together on the subject split: BM25 0.087, TF-IDF 0.112, zero-shot E5-large 0.118, fine-tuned 0.217, with both enrichments 0.255, add the BGE reranker 0.292, and finally blending the reranker and retriever scores gets 0.311. That last number is out-of-fold: I choose the blend weight on one half of the validation questions and score it on the other half. So the full pipeline scores almost three times the best lexical baseline. As for the ceiling, 78 percent of queries have the right misconception somewhere in the top 25, so the reranker can only fix that part. A caution on the reranker: its validation score moved by about 0.02 between epochs, peaking at epoch two and ending lower, so the roughly 0.03 gain from reranking is real in direction but not precise.

## slide 13. what did not help
on slide:
- title: Negative results
- four cards:
  - unseen-label bonus: 0.3085 vs 0.3110 without (out-of-fold)
  - reinforcement-learning selector: REINFORCE 0.312 = supervised policy 0.317 = plain score fusion 0.311
  - MiniLM reranker on the large retriever: 0.308 vs 0.310
  - misconception explanations alone, after fine-tuning: +0.011, within noise

script:
Several things I tried did not help, and I think they are worth reporting. First, a bonus for unseen misconceptions, the trick some competition teams used to correct for their test distribution. Out of fold it adds nothing: 0.3085 versus 0.3110 without it. Second, the reinforcement-learning selector from my proposal. I implemented it: a small policy network scores the 25 candidates, a top-three list is sampled, and it is rewarded with average precision. It reaches 0.312, the same as the supervised listwise version at 0.317, and the same as plain score fusion at 0.311. So the reinforcement learning contributes nothing beyond learning to blend two scores, and with exactly one right answer there is little else for it to add. Third, the small MiniLM reranker did not help the large retriever. And fourth, the misconception explanations on their own did not clearly help after fine-tuning. I would rather show these than hide them, because they tell you where not to spend effort.

## slide 14. limitations
on slide:
- title: What these numbers do not show
- six short cards: single seed, no confidence intervals | 655 and 852 validation rows | validation set also used for reporting (epochs fixed in advance, no checkpoint picking) | generated text from a 7B model, not audited | not comparable to the leaderboard (hidden test, bigger models) | reranker negatives come from a retriever trained on the same rows

script:
Limitations. This is a single seed with small validation sets, 655 and 852 rows, so only large gaps should be trusted. The validation split is also the one I report on, although I fixed the number of epochs in advance and did not select checkpoints. The generated explanations and reasoning come from a seven-billion-parameter model and I have not audited them. The official test set is not available, so I cannot compare to the leaderboard, and I want to be clear that my scores are not comparable to it: the winners faced a harder, mostly-unseen test set and used much bigger models. Finally, the reranker's training negatives come from a retriever that was trained on the same rows, which probably makes them easier than what it will see at test time.

## slide 15. future work
on slide:
- title: Next steps
- numbered list: 1 language-model rerankers (Qwen 14B with LoRA) | 2 synthetic questions for the 983 untrained misconceptions | 3 more seeds, confidence intervals, cross-validated subject splits | 4 audit the generated explanations | 5 teacher-in-the-loop: confirmed labels grow the training set

script:
Future work, in the order I expect it to pay off. One: language-model rerankers, which is where the winners made their biggest jump; even a fourteen-billion-parameter Qwen with LoRA is feasible on one large GPU. Two: synthetic training questions aimed at the 983 misconceptions with no training example, which several top teams said mattered most for unseen labels. Three: more seeds and confidence intervals so smaller gains can be established, plus a cross-validated split by subject. Four: measure the quality of the generated explanations, for instance by checking them against held-out labelled examples. And longer term, the system could suggest misconceptions to a teacher who confirms them, so that confirmed labels grow the training set, which attacks the long tail directly.

## slide 16. takeaways
on slide:
- title: Four takeaways
- 1 retrieve-then-rerank works, but only after fine-tuning: zero-shot is no better than keyword search
- 2 validation design matters as much as the model: subject split = 58% unseen labels, 8 points lower
- 3 offline LLM enrichment gave the biggest gain after fine-tuning: +0.038 MAP@3, +7 points recall
- 4 a large reranker adds ~0.03; RL and unseen bonuses add nothing beyond score blending

script:
To summarise. One: framing misconception tagging as retrieve-then-rerank works, and fine-tuning is what makes it work, since zero-shot models are no better than keyword search. Two: validation design matters as much as the model. Grouping by subject gives 58 percent unseen labels and scores about eight points lower than grouping by question. Three: offline language-model enrichment of both sides, misconception explanations and per-answer reasoning, gave the largest improvement after fine-tuning: 0.038 MAP at 3 and seven points of recall, with no inference cost for the explanations. Four: a larger reranker adds about 0.03, a small one does not, and neither reinforcement learning nor unseen-label bonuses added anything on top of simple score blending.

## slide 17. what I learned (dark, closing)
on slide:
- title: What I learned
- three short lines: most of the work is not the model (labels, split, checking my own pipeline) | report measured numbers only, say when a gap is within noise | decide the split and the noise level first
- repo: github.com/goog-msft-fb-nflx-nvda-aapl/eedi-misconception-retrieval
- thank you

script:
Finally, what I learned. As a data mining project, the biggest lesson was how much of the work is not the model: reading the label distribution, deciding the split, and checking my own pipeline for mistakes. Early on I found that my exploratory analysis reported an imbalance figure that was actually Gini impurity, not a Gini coefficient, so I fixed it before using it, and the data description in my original proposal did not match the dataset I was actually using, so I corrected the framing. I also learned to report only measured numbers and to say when a difference is within noise. If I did it again I would decide the split and the noise level on day one. Thank you for watching. The code, the logs and the full results table are in the GitHub repository shown on the slide, and I am looking forward to your feedback.

## slide 18. references (no narration)
on slide, small text:
- Karpukhin et al. 2020. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- Wang et al. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
- Nogueira and Cho. 2019. Passage Re-ranking with BERT. arXiv:1901.04085.
- Chen et al. 2024. BGE M3-Embedding. arXiv:2402.03216 (BGE reranker v2 m3).
- Lewis et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Qwen Team. 2024. Qwen2.5 Technical Report. arXiv:2412.15115.
- Eedi. 2024. Mining Misconceptions in Mathematics (Kaggle competition) and its top-5 public writeups (competitor reports, not peer reviewed). kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics
- code, logs, results: github.com/goog-msft-fb-nflx-nvda-aapl/eedi-misconception-retrieval

script: none, reference slide.
