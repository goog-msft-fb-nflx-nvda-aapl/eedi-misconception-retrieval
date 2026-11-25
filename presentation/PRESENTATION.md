# presentation brief: eedi misconception retrieval

purpose: build the slides in canva and record the video. the slides are the demo (diagram, charts, big numbers). the speaker script carries every detail.

## the rules for the slides (most important)
- concise and catchy, not verbose. each slide = one punchy headline (8 words max) + one visual + at most one short caption.
- no bullet lists, no paragraphs. if a slide has more than ~15 words besides the headline, cut it. everything else goes in the speaker notes.
- the visuals are the content: use the ready-made figures in `figures/` (png and svg, white background) and the pipeline diagram `figures/f01_pipeline.png`. redraw a figure in canva only if it needs restyling; never change its numbers.
- 19 slides: 18 narrated + 1 references slide. paste each slide's script into its speaker notes, verbatim.
- look: school maths / chalkboard feel. dark green (1F4D45) and charcoal (1C2B2A) for the title and closing slides, white content slides, one amber accent (F2A33A), teal (2A9D8F) secondary. serif for headlines, clean sans for body. headlines 36pt or larger.
- no accent lines under titles, no edge stripes, no stock-photo filler.

## files in this package
- `presentation/PRESENTATION.md`: this brief and the full speaker script
- `figures/f01..f11`: charts and the pipeline diagram (png for canva, svg for editing); `figures/figure_data.json` has every plotted value
- `results/`: raw results; `results/ablation.md` is the full table; `results/error_examples.md` has four real validation examples
- `WORKLOG.md`: every number and setup detail; `SURVEY.md`: related work notes
- code: `0*.py`, `1*.py`, `utils/`

speaker script length: about 2,900 words, roughly 20 to 24 minutes at a calm pace (course asks 20 to 30).
single seed, 655 or 852 validation rows: every claim in the script is worded to match that.

---

## slide 1
headline: Retrieving the Misconception Behind a Wrong Answer
visual: (title slide) subtitle: a small-budget retrieve-and-rerank study on Eedi data. James Christian, Data Mining, CU Boulder.
caption: faint background motif: 0.355 > 0.8 ?

speaker notes (verbatim):
Hello, I'm James Christian, and this is my final project for Data Mining. The title is "Retrieving the Misconception Behind a Wrong Answer", a small-budget retrieve-and-rerank study on the Eedi data. In the next twenty or so minutes I will explain the problem, show the data and what it looks like, describe the pipeline I built, and then walk through the results, including the things that did not work. One thing I want to be upfront about: the task and the data come from the Kaggle Eedi competition from 2024. I am not claiming a new method, and I am not comparing myself to the leaderboard. What I can offer is a controlled, reproducible look at what a modest budget buys you on this problem, with every number measured by me and all the code and logs in a public repository.

## slide 2
headline: A wrong answer is a clue
visual: simple 3-step graphic (draw it): wrong answer 0.355 > 0.8  ->  misconception 'longer number is bigger'  ->  targeted feedback
caption: Goal: the right misconception in the top 3

speaker notes (verbatim):
Let me start with the problem. Students aged nine to sixteen often develop systematic wrong ways of thinking, called misconceptions. A classic example: a student decides that 0.355 is bigger than 0.8, because 355 is bigger than 8. They are applying whole-number intuition to decimals. If a teacher knows the student has that misconception, the feedback can be targeted: here is why decimals do not work like that. The trouble is that working out the misconception from a wrong answer is slow, and it does not scale across a classroom of thirty students and many questions. So the data mining question is this. Given a multiple-choice maths question, the correct answer, and the wrong answer a student picked, can we automatically rank which misconception most likely explains that wrong answer? I frame it as ranked retrieval rather than flat classification, because there are over two and a half thousand possible misconceptions, and many of them are rare or never appear in the training data. The goal is that the right misconception shows up in the top three.

## slide 3
headline: 4,370 wrong answers. 2,587 misconceptions.
visual: 3 big numbers: 1,869 questions | 4,370 queries | 2,587 misconceptions. optional: one example card, 'Believes there are 100 degrees in a full turn'
caption: Source: Kaggle Eedi 2024

speaker notes (verbatim):
Here is the data. It comes from the Eedi competition on Kaggle. Each row is a question with four options, one correct and up to three wrong. For every wrong option an expert has labelled the misconception behind it, chosen from a fixed taxonomy of 2,587 misconceptions. After reshaping the table into one row per wrong answer I have 4,370 queries from 1,869 questions, about 2.3 wrong answers per question. The query I give the model is the construct and subject names, the question text, the correct answer and the wrong answer, joined into one string. The target is a short sentence, for example "Uses dividing fractions method for multiplying fractions", or "Believes there are 100 degrees in a full turn". The official test file in this competition is only a three-row stub, so all my evaluation is on held-out parts of the training data. I will come back to how I split it, because that turns out to matter a lot.

## slide 4
headline: 38% of labels were never trained on
visual: figures/f02_label_longtail.png
caption: 983 never | 747 once | 857 two or more

speaker notes (verbatim):
Before modelling I looked at the labels, because they decide what is achievable. Of the 2,587 misconceptions in the taxonomy, 983 never occur as a label in the training file at all. Another 747 occur exactly once, and only 857 occur two or more times. The Gini coefficient of the label counts over the whole taxonomy is 0.67, so the distribution is heavily long-tailed, although the biggest single label is only 54 rows and the ten most common together are under 8 percent of the data. Two consequences. First, a classifier over seen labels can never output 38 percent of the taxonomy. That is why I use retrieval: the misconception text is an input, so in principle an unseen misconception can still be matched. Second, how I split validation decides whether I measure that ability or hide it. The texts themselves are short: questions average about 29 words and misconception names about 12.

## slide 5
headline: Same recipe as the winners, 100x smaller
visual: 3-column graphic (draw it): building blocks (DPR, E5, cross-encoders) | Eedi winners (Qwen 14B to 72B, synthetic data) | this project (0.03 to 0.6B, one GPU)
caption: Controlled ablation, not a leaderboard run

speaker notes (verbatim):
Related work comes from three places. Methodologically, dense passage retrieval showed that you can encode queries and passages independently with a bi-encoder and search by similarity. E5 gives strong general-purpose embeddings. Cross-encoder rerankers, such as Nogueira and Cho's BERT reranking and the BGE reranker, score a query and a candidate jointly, which is slower but more precise. Retrieval-augmented generation popularised the retrieve-then-use pattern. Second, the Eedi competition itself. I read the top five public writeups. The winners used the same two-stage shape but with much bigger models: Qwen 2.5 retrievers and rerankers from 14 to 72 billion parameters, around ten thousand synthetic training questions, and large-model reasoning added to the query. Two teams also had language models write explanations for each misconception. Several reported that the hidden test set was dominated by misconceptions unseen in training, and one said that validating by subject tracked the leaderboard better than validating by question. Those are competitor reports, not peer-reviewed results. Third, my contribution relative to that is a scaled-down version: models between about 0.03 and 0.6 billion parameters on one GPU, run as a controlled ablation instead of a leaderboard attempt.

## slide 6
headline: Retrieve, then rerank
visual: figures/f01_pipeline.png
caption: Orange = offline LLM enrichment, no labels

speaker notes (verbatim):
Here is the pipeline. A query goes into a bi-encoder, multilingual E5, which embeds it. The 2,587 misconceptions are embedded once. I take the top 25 by cosine similarity. A cross-encoder reranker then reads the query together with each of the 25 candidates and re-scores them, and the top three are my answer. For training, the bi-encoder uses a contrastive loss with in-batch negatives plus one BM25 hard negative per query, and I mask out cases where two items in a batch share the same misconception, since those are not true negatives. The reranker uses a listwise softmax loss over one positive and fifteen retrieved negatives. On top of that, two offline enrichments using Qwen2.5 7B Instruct. One: expand each misconception with a two-sentence explanation, which costs nothing at inference time. Two: generate a short reasoning for each wrong answer, from the question, correct answer and wrong answer only. Neither uses the labels. I also test a reinforcement-learning selector, which I will come to.

## slide 7
headline: Two ways to split. One is honest.
visual: figures/f03_unseen_share.png + a small chip 'MAP@3: right answer in the top 3'
caption: Subject split = the realistic one

speaker notes (verbatim):
How I evaluate. The main metric is mean average precision at three. Because each query has exactly one right misconception, you get full credit for rank one, a half for rank two, a third for rank three, and nothing otherwise. I also report MAP at 25 and recall at 25, because recall at 25 is the ceiling the reranker works under. I use two validation splits, each 15 percent, grouped so related rows stay together. The question split keeps all rows of one question on one side. The subject split holds out whole curriculum topics. The difference is large. On the question split, 26 percent of validation rows carry a misconception that never appears in training. On the subject split it is 58 percent. I treat the subject split as the realistic one. Everything is a single seed with 655 or 852 validation rows, and I train for a fixed number of epochs and report the last one, without picking the best checkpoint on validation. So I treat differences below about 0.02 MAP at 3 as not established.

## slide 8
headline: Zero-shot fails. Fine-tuning triples it.
visual: figures/f04_baselines_finetune_qs.png
caption: Recall@25 = 0.81 for E5-large fine-tuned

speaker notes (verbatim):
First results, on the question split. The lexical baselines are weak: BM25 gets 0.089 MAP at 3, TF-IDF 0.110. Zero-shot E5, without any training, is no better: 0.061 for the small model and 0.100 for the large one. That matches what the competition organisers said, that pre-trained models did not work out of the box. Fine-tuning changes the picture. Small E5 goes to 0.270 and large E5 to 0.310. That is roughly a threefold improvement over zero-shot, and recall at 25 for the large model is 0.81, meaning the right misconception is in the candidate list for four of five queries. So the domain gap is real but learnable: with only about 3,700 training pairs, the encoder learns what the similarity between a wrong answer and its misconception looks like.

## slide 9
headline: A reranker must be big enough
visual: figures/f05_reranker_qs.png
caption: MiniLM: no help on E5-large. BGE: +0.03

speaker notes (verbatim):
Next, the reranker, on the same question split. With the small retriever, adding a MiniLM cross-encoder improves MAP at 3 from 0.270 to 0.291. With the large retriever, the same MiniLM reranker does not help: 0.308 versus 0.310. So a small general reranker adds nothing once the retriever is stronger. Switching to a larger multilingual reranker, BGE reranker v2 m3 at around 570 million parameters, gives 0.339, a gain of about 0.03 over the retriever alone. My reading, and I label it as interpretation, is that the reranker needs enough capacity to separate near-duplicate misconceptions, which is what the competition winners described when they moved to big language-model rankers. I have not tested that explanation directly.

## slide 10
headline: New topics cost 8 points
visual: figures/f06_split_gap.png
caption: Same model, different split

speaker notes (verbatim):
Now the realistic split. Same model, same training recipe: the large E5 retriever fine-tuned on misconception names only. It scores 0.310 MAP at 3 on the question split and 0.232 on the subject split. It loses eight points, because on the question split it has seen other questions on the same topics during training. This is why the competition teams argued about validation. If I had only reported the question split, I would be overstating how well this works on new topics. One encouraging detail: on the subject split, recall at 25 for rows with unseen misconceptions is 0.75, not lower than the 0.67 for seen ones, so the retrieval formulation does reach labels the model never trained on. The unseen rows are different questions, so I would not read too much into the direction of that gap. I use the subject split for everything that follows.

## slide 11
headline: LLM enrichment: +0.038
visual: figures/f07_enrichment_ss.png
caption: Explanations + reasoning, both offline

speaker notes (verbatim):
Now the language-model enrichments, all on the subject split and at the same input length, so the comparison is clean. Names only: 0.217 MAP at 3 and a recall at 25 of 0.71. Adding the explanations to the misconceptions: 0.228 and 0.74. Adding the generated reasoning to the query: 0.222 and 0.75. Both together: 0.255 and 0.78. Each addition alone is within the noise I measured, about 0.015 just from changing the input length, but both lift recall, and together the gain of 0.038 MAP at 3 and seven points of recall is larger than that noise. Zero-shot, the explanations matter most: recall at 25 jumps from 0.42 to 0.58 with no training at all. That fits what the top Eedi teams reported, here at the scale of a seven-billion-parameter generator. One caveat: the generated text is uneven. Some explanations are muddled, and I have not measured their quality.

## slide 12
headline: From keyword search to 0.31
visual: figures/f08_ladder_ss.png
caption: Almost 3x the best lexical baseline

speaker notes (verbatim):
Putting it together on the subject split: BM25 0.087, TF-IDF 0.112, zero-shot E5-large 0.118, fine-tuned 0.217, with both enrichments 0.255, add the BGE reranker 0.292, and finally blending the reranker and retriever scores gets 0.311. That last number is out-of-fold: I choose the blend weight on one half of the validation questions and score it on the other half. So the full pipeline scores almost three times the best lexical baseline. As for the ceiling, 78 percent of queries have the right misconception somewhere in the top 25, so the reranker can only fix that part. A caution on the reranker: its validation score moved by about 0.02 between epochs, peaking at epoch two and ending lower, so the roughly 0.03 gain from reranking is real in direction but not precise.

## slide 13
headline: Where the right answer lands
visual: figures/f09_rank_of_correct.png (+ optional: one real example card from figures/../results/error_examples.md)
caption: 42% in the top 3. 78% in the top 25.

speaker notes (verbatim):
To make this concrete, here is where the correct misconception lands for the 852 validation queries. For 24 percent it is ranked first, and for another 18 percent it is second or third, so 42 percent are in the top three. For 37 percent it sits somewhere between fourth and twenty-fifth, and for 22 percent it is not in the top 25 at all, which is the retrieval ceiling I mentioned. Top-three accuracy is 45 percent for seen labels and 39 percent for unseen ones. The examples I picked by a fixed rule, the first validation row in each category, are instructive. In the area-units question, when the student wrote metres, the model ranks 'Does not know units of area should be squared' first. For another wrong answer to the same question, cubic kilometres, the right misconception is third, behind 'Believes squared units can be used to measure lengths' and a volume misconception. Those are plausible neighbours, and the taxonomy is full of near-duplicates, which is why the top three is hard to hit. And a miss: for a graph question where the student said the dog is accelerating, the true label is 'Believes the gradient of a distance-time graph is the acceleration', and the model's top three are other graph-slope misconceptions. Close, but not the same.

## slide 14
headline: What didn't work
visual: figures/f10_posthoc_final.png + 3 short chips: unseen bonus | RL selector | small reranker on large retriever
caption: RL = supervised = plain blending

speaker notes (verbatim):
Several things I tried did not help, and I think they are worth reporting. First, a bonus for unseen misconceptions, the trick some competition teams used to correct for their test distribution. Out of fold it adds nothing: 0.3085 versus 0.3110 without it. Second, the reinforcement-learning selector from my proposal. I implemented it: a small policy network scores the 25 candidates, a top-three list is sampled, and it is rewarded with average precision. It reaches 0.312, the same as the supervised listwise version at 0.317, and the same as plain score fusion at 0.311. So the reinforcement learning contributes nothing beyond learning to blend two scores, and with exactly one right answer there is little else for it to add. Third, the small MiniLM reranker did not help the large retriever. And fourth, the misconception explanations on their own did not clearly help after fine-tuning. I would rather show these than hide them, because they tell you where not to spend effort.

## slide 15
headline: Handle with care
visual: 6 small icons or chips (draw them): single seed | small validation sets | not leaderboard-comparable | generated text unaudited | val also used for reporting | easy reranker negatives
caption: Differences under 0.02 are noise

speaker notes (verbatim):
Limitations. This is a single seed with small validation sets, 655 and 852 rows, so only large gaps should be trusted. The validation split is also the one I report on, although I fixed the number of epochs in advance and did not select checkpoints. The generated explanations and reasoning come from a seven-billion-parameter model and I have not audited them. The official test set is not available, so I cannot compare to the leaderboard, and I want to be clear that my scores are not comparable to it: the winners faced a harder, mostly-unseen test set and used much bigger models. Finally, the reranker's training negatives come from a retriever that was trained on the same rows, which probably makes them easier than what it will see at test time.

## slide 16
headline: Next: bigger rankers, more data
visual: numbered roadmap graphic (draw it): 1 LLM rerankers | 2 synthetic questions for unseen labels | 3 more seeds | 4 audit generated text | 5 teacher in the loop
caption: Where the winners made their jumps

speaker notes (verbatim):
Future work, in the order I expect it to pay off. One: language-model rerankers, which is where the winners made their biggest jump; even a fourteen-billion-parameter Qwen with LoRA is feasible on one large GPU. Two: synthetic training questions aimed at the 983 misconceptions with no training example, which several top teams said mattered most for unseen labels. Three: more seeds and confidence intervals so smaller gains can be established, plus a cross-validated split by subject. Four: measure the quality of the generated explanations, for instance by checking them against held-out labelled examples. And longer term, the system could suggest misconceptions to a teacher who confirms them, so that confirmed labels grow the training set, which attacks the long tail directly.

## slide 17
headline: Four takeaways
visual: 4 icon cards (draw them): fine-tuning is what works | the split is half the result | LLM enrichment +0.038 | RL adds nothing
caption: One number each: 3x | 58% unseen | +0.038 | +0.00

speaker notes (verbatim):
To summarise. One: framing misconception tagging as retrieve-then-rerank works, and fine-tuning is what makes it work, since zero-shot models are no better than keyword search. Two: validation design matters as much as the model. Grouping by subject gives 58 percent unseen labels and scores about eight points lower than grouping by question. Three: offline language-model enrichment of both sides, misconception explanations and per-answer reasoning, gave the largest improvement after fine-tuning: 0.038 MAP at 3 and seven points of recall, with no inference cost for the explanations. Four: a larger reranker adds about 0.03, a small one does not, and neither reinforcement learning nor unseen-label bonuses added anything on top of simple score blending.

## slide 18
headline: What I learned
visual: 3 short lines + repo link
caption: github.com/goog-msft-fb-nflx-nvda-aapl/eedi-misconception-retrieval

speaker notes (verbatim):
Finally, what I learned. As a data mining project, the biggest lesson was how much of the work is not the model: reading the label distribution, deciding the split, and checking my own pipeline for mistakes. Early on I found that my exploratory analysis reported an imbalance figure that was actually Gini impurity, not a Gini coefficient, so I fixed it before using it, and the data description in my original proposal did not match the dataset I was actually using, so I corrected the framing. I also learned to report only measured numbers and to say when a difference is within noise. If I did it again I would decide the split and the noise level on day one. Thank you for watching. The code, the logs and the full results table are in the GitHub repository shown on the slide, and I am looking forward to your feedback.

## slide 19 (references, no narration)
headline: References
small text:
- Karpukhin et al. 2020. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- Wang et al. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.
- Nogueira and Cho. 2019. Passage Re-ranking with BERT. arXiv:1901.04085.
- Chen et al. 2024. BGE M3-Embedding. arXiv:2402.03216 (BGE reranker v2 m3).
- Lewis et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Qwen Team. 2024. Qwen2.5 Technical Report. arXiv:2412.15115.
- Eedi. 2024. Mining Misconceptions in Mathematics (Kaggle) and its top-5 public writeups (competitor reports, not peer reviewed). kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics
- code, logs, results: github.com/goog-msft-fb-nflx-nvda-aapl/eedi-misconception-retrieval
