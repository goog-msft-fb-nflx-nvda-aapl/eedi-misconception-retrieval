| config                                         | split      |   MAP@3 |   MAP@25 |   Recall@25 (candidates) |
|:-----------------------------------------------|:-----------|--------:|---------:|-------------------------:|
| BM25                                           | QuestionId |  0.0885 |   0.1134 |                   0.371  |
| TF-IDF+cosine                                  | QuestionId |  0.1099 |   0.1404 |                   0.4229 |
| Zero-shot E5-small                             | QuestionId |  0.0611 |   0.0867 |                   0.3008 |
| BM25                                           | SubjectId  |  0.0869 |   0.118  |                   0.3826 |
| TF-IDF+cosine                                  | SubjectId  |  0.1125 |   0.1516 |                   0.466  |
| Zero-shot E5-small                             | SubjectId  |  0.0826 |   0.1067 |                   0.3286 |
| e5l zero-shot (epoch 0)                        | QuestionId |  0.1    |   0.1287 |                   0.3847 |
| e5l fine-tuned                                 | QuestionId |  0.3099 |   0.3636 |                   0.8092 |
| e5l_subj zero-shot (epoch 0)                   | SubjectId  |  0.1164 |   0.1481 |                   0.4237 |
| e5l_subj fine-tuned                            | SubjectId  |  0.232  |   0.2816 |                   0.7136 |
| e5l_subj_L320 zero-shot (epoch 0)              | SubjectId  |  0.118  |   0.1492 |                   0.4214 |
| e5l_subj_L320 fine-tuned                       | SubjectId  |  0.2171 |   0.2665 |                   0.7101 |
| e5l_subj_exp zero-shot (epoch 0)               | SubjectId  |  0.1371 |   0.1813 |                   0.5763 |
| e5l_subj_exp fine-tuned                        | SubjectId  |  0.2248 |   0.2807 |                   0.7312 |
| e5l_subj_exp_L320 zero-shot (epoch 0)          | SubjectId  |  0.1371 |   0.1813 |                   0.5763 |
| e5l_subj_exp_L320 fine-tuned                   | SubjectId  |  0.2285 |   0.2834 |                   0.7394 |
| e5l_subj_expqr_L320 zero-shot (epoch 0)        | SubjectId  |  0.141  |   0.1877 |                   0.5739 |
| e5l_subj_expqr_L320 fine-tuned                 | SubjectId  |  0.2553 |   0.3115 |                   0.7817 |
| e5l_subj_qr_L320 zero-shot (epoch 0)           | SubjectId  |  0.1227 |   0.1535 |                   0.4261 |
| e5l_subj_qr_L320 fine-tuned                    | SubjectId  |  0.2218 |   0.2772 |                   0.7547 |
| e5s zero-shot (epoch 0)                        | QuestionId |  0.0606 |   0.0864 |                   0.3038 |
| e5s fine-tuned                                 | QuestionId |  0.2695 |   0.325  |                   0.7634 |
| e5l_bge reranker over its retriever            | QuestionId |  0.3394 |   0.3925 |                   0.8076 |
| e5l_mini reranker over its retriever           | QuestionId |  0.3079 |   0.3609 |                   0.8076 |
| e5l_subj_bge reranker over its retriever       | SubjectId  |  0.2637 |   0.3091 |                   0.7101 |
| e5l_subj_expqr_bge reranker over its retriever | SubjectId  |  0.2917 |   0.3461 |                   0.7817 |
| mini reranker over its retriever               | QuestionId |  0.2913 |   0.3383 |                   0.7618 |