"""
06_expand_misconceptions.py
Offline LLM expansion of each misconception name (input = name only, no labels / no train data).
Output: data/misconception_expanded.csv  (MisconceptionId, MisconceptionName, Expansion)
Run:  python 06_expand_misconceptions.py --model Qwen/Qwen2.5-7B-Instruct
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helpers import DATA_DIR

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--max_new", type=int, default=90)
a = p.parse_args()

mis = pd.read_csv(DATA_DIR / "misconception_mapping.csv")
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
SYS = ("You are a maths teacher. You are given a misconception that students aged 9-16 may have. "
       "In 2 sentences, explain what the student is thinking and give one short example of a wrong answer it leads to. "
       "Do not give the correct method.")
outs = []
names = mis["MisconceptionName"].tolist()
for i in range(0, len(names), a.bs):
    msgs = [tok.apply_chat_template([{"role": "system", "content": SYS},
                                     {"role": "user", "content": f"Misconception: {n}"}],
                                    tokenize=False, add_generation_prompt=True) for n in names[i:i + a.bs]]
    enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
    outs += [t.strip().replace("\n", " ") for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    print(i + len(msgs), "/", len(names), flush=True)
mis["Expansion"] = outs
mis.to_csv(DATA_DIR / "misconception_expanded.csv", index=False)
print(mis.head(3).to_string())
