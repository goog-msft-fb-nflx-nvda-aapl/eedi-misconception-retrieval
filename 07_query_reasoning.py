"""
07_query_reasoning.py
Offline LLM reasoning per (question, wrong answer) row: why might a student pick this wrong answer?
Input: question, correct answer, wrong answer. NO misconception label, no train supervision.
Output: data/query_reasoning.csv  (QuestionId, WrongAnswer, Reasoning)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helpers import DATA_DIR, load_data

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--bs", type=int, default=48)
p.add_argument("--max_new", type=int, default=110)
a = p.parse_args()

df, _, _ = load_data()
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
SYS = ("You are a maths teacher. A student answered a multiple-choice question incorrectly. "
       "In at most 3 sentences, describe the likely mistake in the student's thinking that leads to the wrong answer. "
       "Do not give the correct method.")
def prompt(r):
    u = (f"Question: {r.QuestionText}\nCorrect answer: {r.CorrectAnswerText}\nStudent's wrong answer: {r.WrongAnswerText}")
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                   tokenize=False, add_generation_prompt=True)
rows = list(df.itertuples()); outs = []
for i in range(0, len(rows), a.bs):
    enc = tok([prompt(r) for r in rows[i:i + a.bs]], return_tensors="pt", padding=True, truncation=True, max_length=900).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
    outs += [t.strip().replace("\n", " ") for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    print(len(outs), "/", len(rows), flush=True)
out = pd.DataFrame({"QuestionId": df.QuestionId, "WrongAnswer": df.WrongAnswer, "Reasoning": outs})
out.to_csv(DATA_DIR / "query_reasoning.csv", index=False)
print(out.head(3).to_string())
