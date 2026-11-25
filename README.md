# eedi misconception retrieval

scaled-down retrieve-then-rerank study on the kaggle eedi "mining misconceptions in mathematics" data (2024).
task: given a multiple-choice math question, the correct answer and one wrong answer, rank the misconception (out of 2,587) behind the wrong answer.

this is a controlled small-budget study (e5 bi-encoders, minilm / bge rerankers, one gpu), not a new method and not a leaderboard entry.
the winning teams used 14b-72b qwen rankers and synthetic data; see SURVEY.md.

## what is in here
- `01_eda.py`, `01b_label_stats.py`: data and label statistics
- `02_baselines.py`: bm25, tf-idf, zero-shot e5
- `03_train_retriever.py`: bi-encoder fine-tuning (infonce, in-batch + bm25 hard negative, false-negative masking, optional `--expand`)
- `04_train_reranker.py`: listwise cross-encoder reranker over the retriever top-25
- `06_expand_misconceptions.py`: offline llm explanation of each misconception name (qwen2.5-7b-instruct)
- `07_query_reasoning.py`: offline llm reasoning per (question, wrong answer), no label used
- `08_calibration.py`: score fusion and unseen-label bonus, tuned on one half of val and scored on the other
- `09_rl_selector.py`: reinforce top-3 selector vs a supervised listwise policy on the same features
- `05_make_tables.py`: builds `results/ablation.md` from the saved histories
- `results/`: per-epoch histories (json), baseline csvs, label stats
- `WORKLOG.md`: every number, the setup behind it and what was not done
- `SURVEY.md`: what the top eedi teams did (competitor writeups, not peer reviewed)

## splits
`SPLIT_BY=QuestionId` (default) or `SPLIT_BY=SubjectId`, 15% grouped validation, seed 42.
the subject split is the harder one: 58% of its validation rows have a misconception never seen in train (26% for the question split).

## run
```
./download_data.sh
python 01_eda.py && python 01b_label_stats.py
python 02_baselines.py
python 03_train_retriever.py --model_name intfloat/multilingual-e5-large --tag e5l --lr 1.5e-5
python 04_train_reranker.py --retriever_tag e5l --reranker_model BAAI/bge-reranker-v2-m3 --lr 1e-5 --tag e5l_bge
SPLIT_BY=SubjectId python 03_train_retriever.py --model_name intfloat/multilingual-e5-large --tag e5l_subj --lr 1.5e-5
python 06_expand_misconceptions.py && python 07_query_reasoning.py   # then add --expand / --qreason to 03 and 04
SPLIT_BY=SubjectId python 08_calibration.py --scores models/valscores_<tag>.npz --tag subj
SPLIT_BY=SubjectId python 09_rl_selector.py --scores models/valscores_<tag>.npz --tag subj
python 05_make_tables.py
```
the kaggle data, model weights and the generated expansions are not in the repo.

## results
see `results/ablation.md` and `WORKLOG.md`. single seed, 655 / 852 validation rows: differences under about 0.02 map@3 are not established.
