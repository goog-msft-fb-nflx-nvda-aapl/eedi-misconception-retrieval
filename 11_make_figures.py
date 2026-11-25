"""
11_make_figures.py
Presentation figures (png + svg) built only from the saved results/*.json|csv. No models are loaded.
Run: python 11_make_figures.py     -> figures/*.png, figures/*.svg, figures/figure_data.json
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results"); OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
GREEN, DARK, TEAL, AMBER, CORAL, GREY, PALE = "1F4D45", "1C2B2A", "2A9D8F", "F2A33A", "E76F51", "B8C4C0", "EEF3F1"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": "#" + GREY, "axes.labelcolor": "#" + DARK, "text.color": "#" + DARK,
                     "xtick.color": "#" + DARK, "ytick.color": "#" + DARK, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "svg.fonttype": "none"})
J = lambda f: json.load(open(os.path.join(R, f)))
hist = lambda tag: J(f"hist_ret_{tag}.json")["history"]
rr = lambda tag: J(f"hist_rr_{tag}.json")["history"][-1]["MAP@3"]
base_qs = pd.read_csv(os.path.join(R, "baseline_results.csv")).set_index("model")
base_ss = pd.read_csv(os.path.join(R, "baseline_results_SubjectId.csv")).set_index("model")
data = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hbar(name, labels, vals, colors, title, xlim, note=None):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    y = range(len(labels))[::-1]
    bars = ax.barh(list(y), vals, color=["#" + c for c in colors], height=0.66)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.set_xlim(0, xlim); ax.set_xlabel("MAP@3")
    for b, v in zip(bars, vals):
        ax.text(v + xlim * 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=17, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold")
    if note: fig.text(0.01, -0.02, note, fontsize=13, color="#555", ha="left")
    data[name] = dict(zip(labels, [round(v, 4) for v in vals])); save(fig, name)


def col(name, labels, vals, colors, title, ylim, ylabel="MAP@3", note=None, figsize=(12, 6.2), rot=0):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, vals, color=["#" + c for c in colors], width=0.62)
    ax.set_ylim(0, ylim); ax.set_ylabel(ylabel)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ylim * 0.012, f"{v:.3f}" if v < 1.5 else f"{v:,.0f}", ha="center", fontsize=17, fontweight="bold")
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot else "center")
    if note: fig.text(0.01, -0.03, note, fontsize=13, color="#555", ha="left")
    data[name] = dict(zip(labels, [round(v, 4) for v in vals])); save(fig, name)


# f01 pipeline diagram ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6.4)); ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis("off")
def box(x, y, w, h, text, fc, tc="white", fs=16, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.18", fc="#" + fc, ec="#" + (ec or fc), lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", linespacing=1.3)
def arrow(x1, y1, x2, y2, c=DARK, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#" + c, linestyle=ls))
box(0.2, 2.4, 2.6, 1.6, "Query\nquestion + correct answer\n+ wrong answer", PALE, DARK, 14, GREY)
box(3.5, 2.4, 2.6, 1.6, "E5 bi-encoder\n(fine-tuned)", GREEN, fs=16)
box(6.8, 2.4, 2.0, 1.6, "Top 25\ncandidates", TEAL, fs=16)
box(9.4, 2.4, 2.4, 1.6, "BGE cross-encoder\nreranker\n(listwise)", GREEN, fs=15)
box(12.2, 2.4, 1.6, 1.6, "Top 3\nanswer", AMBER, DARK, 16)
for a, b in [(2.8, 3.5), (6.1, 6.8), (8.8, 9.4), (11.8, 12.2)]: arrow(a, 3.2, b, 3.2)
box(3.2, 0.3, 3.2, 1.1, "2,587 misconceptions,\nembedded once", PALE, DARK, 14, GREY); arrow(4.8, 1.4, 4.8, 2.4)
box(0.2, 4.9, 4.7, 1.2, "Offline LLM reasoning per wrong answer\n(Qwen2.5-7B, no labels)", "FCE8C8", DARK, 14, AMBER); arrow(2.6, 4.9, 2.0, 4.0, AMBER, "--")
box(5.4, 4.9, 4.7, 1.2, "Offline LLM explanation per misconception\n(Qwen2.5-7B, name only)", "FCE8C8", DARK, 14, AMBER); arrow(6.4, 4.9, 5.2, 4.0, AMBER, "--")
ax.text(10.6, 5.5, "dashed = offline enrichment\nno labels used", fontsize=13, color="#555", va="center")
save(fig, "f01_pipeline")

# f02 long tail ---------------------------------------------------------------------
ls = J("label_stats.json")
col("f02_label_longtail", ["never in\ntraining", "exactly\n1 row", "2 or more\nrows"],
    [ls["classes_never_in_train_csv"], ls["seen_once"], ls["classes_in_train_csv"] - ls["seen_once"]], [CORAL, AMBER, GREEN],
    "Misconceptions by number of training rows (2,587 total)", 1150, "number of misconceptions",
    note=f"Gini coefficient over the taxonomy {ls['gini_coefficient_over_taxonomy']:.2f}; most common label {ls['max_count']} rows; top 10 labels {ls['top10_share_of_rows']*100:.1f}% of rows")

# f03 unseen share per split ----------------------------------------------------------
col("f03_unseen_share", ["question split\n(170 of 655 rows)", "subject split\n(491 of 852 rows)"], [170 / 655 * 100, 491 / 852 * 100], [GREY, CORAL],
    "Validation rows whose misconception never appears in training", 80, "% of validation rows", figsize=(9, 6.2))

# f04 baselines + fine-tuning, question split --------------------------------------------
e5s, e5l = hist("e5s"), hist("e5l")
hbar("f04_baselines_finetune_qs",
     ["BM25", "TF-IDF + cosine", "E5-small zero-shot", "E5-large zero-shot", "E5-small fine-tuned", "E5-large fine-tuned"],
     [base_qs.loc["BM25", "MAP@3"], base_qs.loc["TF-IDF+cosine", "MAP@3"], e5s[0]["MAP@3"], e5l[0]["MAP@3"], e5s[-1]["MAP@3"], e5l[-1]["MAP@3"]],
     [GREY, GREY, CORAL, CORAL, GREEN, GREEN], "Question split: zero-shot fails, fine-tuning roughly triples MAP@3", 0.38)

# f05 reranker, question split ----------------------------------------------------------------
col("f05_reranker_qs", ["E5-small\nfine-tuned", "+ MiniLM\nreranker", "E5-large\nfine-tuned", "+ MiniLM\nreranker", "+ BGE-v2-m3\nreranker"],
    [e5s[-1]["MAP@3"], rr("mini"), e5l[-1]["MAP@3"], rr("e5l_mini"), rr("e5l_bge")], [GREEN, AMBER, GREEN, AMBER, AMBER],
    "Question split: a reranker helps only if it is big enough", 0.40)

# f06 split gap ------------------------------------------------------------------------------------
ss192 = hist("e5l_subj")
col("f06_split_gap", ["question split", "subject split"], [e5l[-1]["MAP@3"], ss192[-1]["MAP@3"]], [GREEN, CORAL],
    "Same model (E5-large, fine-tuned, names only), two splits", 0.40, figsize=(9, 6.2),
    note=f"Recall@25 on unseen labels, subject split: {ss192[-1]['unseen']['Recall@25']:.2f}; on seen labels: {ss192[-1]['seen']['Recall@25']:.2f}")

# f07 enrichment ablation, subject split @320 ------------------------------------------------------------
tags = [("names only", "e5l_subj_L320"), ("+ misconception\nexplanations", "e5l_subj_exp_L320"),
        ("+ answer\nreasoning", "e5l_subj_qr_L320"), ("+ both", "e5l_subj_expqr_L320")]
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
for ax, key, ttl, lim in ((axes[0], "MAP@3", "MAP@3", 0.34), (axes[1], "Recall@25", "Recall@25", 0.95)):
    vals = [hist(t)[-1][key] for _, t in tags]
    bars = ax.bar([l for l, _ in tags], vals, color=["#" + c for c in (GREY, TEAL, TEAL, GREEN)], width=0.65)
    ax.set_ylim(0, lim); ax.set_title(ttl, loc="left", fontsize=20, fontweight="bold")
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + lim * 0.012, f"{v:.3f}", ha="center", fontsize=16, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=13)
    data["f07_" + key] = {l.replace("\n", " "): round(v, 4) for (l, _), v in zip(tags, vals)}
fig.suptitle("Subject split, E5-large fine-tuned, same input length (320 tokens)", x=0.01, ha="left", fontsize=19, fontweight="bold", y=1.02)
save(fig, "f07_enrichment_ss")

# f08 ladder, subject split ------------------------------------------------------------------------------------
cal = J("calibration_subj_expqr.json")
hbar("f08_ladder_ss",
     ["BM25", "TF-IDF + cosine", "E5-large zero-shot", "E5-large fine-tuned", "+ explanations + reasoning", "+ BGE reranker", "+ score fusion (out-of-fold)"],
     [base_ss.loc["BM25", "MAP@3"], base_ss.loc["TF-IDF+cosine", "MAP@3"], hist("e5l_subj_L320")[0]["MAP@3"], hist("e5l_subj_L320")[-1]["MAP@3"],
      hist("e5l_subj_expqr_L320")[-1]["MAP@3"], rr("e5l_subj_expqr_bge"), cal["+ retriever fusion"]["MAP@3_oof"]],
     [GREY, GREY, CORAL, GREEN, GREEN, GREEN, AMBER], "Subject split: from keyword search to 0.31 MAP@3", 0.40)

# f09 rank distribution of the correct misconception -------------------------------------------------------------
ea = J("error_analysis.json")
col("f09_rank_of_correct", ["rank 1", "rank 2 or 3", "rank 4 to 25", "not in\ntop 25"],
    [ea["top1"] * 100, ea["rank2or3"] * 100, ea["rank4to25"] * 100, ea["not_in_top25"] * 100], [GREEN, TEAL, GREY, CORAL],
    "Where the correct misconception lands (final pipeline, subject split)", 50, "% of 852 validation queries")

# f10 post-hoc on final scores ----------------------------------------------------------------------------------------
rl = J("rl_subj_expqr_unseen0.json")
col("f10_posthoc_final", ["reranker\nonly", "+ score\nfusion", "+ fusion +\nunseen bonus", "supervised\nlistwise policy", "REINFORCE\npolicy"],
    [cal["reranker only"]["MAP@3_oof"], cal["+ retriever fusion"]["MAP@3_oof"], cal["+ fusion + unseen bonus"]["MAP@3_oof"],
     rl["supervised_listwise"]["MAP@3_oof_mean"], rl["reinforce"]["MAP@3_oof_mean"]], [GREEN, AMBER, GREY, AMBER, AMBER],
    "Post-hoc on the final scores (out-of-fold): RL adds nothing beyond blending", 0.40, figsize=(13, 6.2))

# f11 training curves ----------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.2))
for (lab, t), c in zip(tags, (GREY, TEAL, AMBER, GREEN)):
    h = hist(t); ax.plot([e["epoch"] for e in h], [e["MAP@3"] for e in h], marker="o", lw=3, color="#" + c, label=lab.replace("\n", " "))
ax.set_xlabel("epoch (0 = zero-shot)"); ax.set_ylabel("MAP@3 (validation)"); ax.legend(frameon=False, fontsize=14)
ax.set_title("Retriever training curves, subject split", loc="left", fontsize=20, fontweight="bold"); save(fig, "f11_training_curves")

json.dump(data, open(os.path.join(OUT, "figure_data.json"), "w"), indent=1)
print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
