"""
Create paper figures from merged_papers_keywords_keybert.xlsx (KeyBERT keywords).
Produces three figure types matching the reference paper:
  Fig A — Word cloud grid per subfield (like Fig 4 & 5)
  Fig B — Subfield mind-map / hierarchy (like Fig 6)
  Fig C — Topic/subfield proportion over time (like Fig 7)
"""

import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_EXCEL = SCRIPT_DIR / "merged_papers_keywords_keybert.xlsx"
OUTPUT_DIR  = SCRIPT_DIR / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("wordcloud not installed – word cloud figures will be skipped.")

# ── Subfield taxonomy (adapted for ML conferences 2017-2025) ──────────────────
SUBFIELDS = {
    "Large Language Models": {
        "color": "#C0392B",
        "patterns": [
            "language model", "large language", "llm", "gpt", "bert", "chatgpt",
            "instruction tun", "prompt", "text generation", "machine translation",
            "question answer", "summarization", "dialogue", "conversational",
            "fine-tun", "pretrain", "pre-train", "natural language", "text classif",
            "sentiment", "named entity", "reading comprehension", "text2", "seq2seq",
            "encoder decoder", "autoregressive", "token", "vocabulary",
        ],
        "topics": [
            "Language Model Pretraining", "Instruction Tuning",
            "Prompt Engineering", "Text Generation", "Machine Translation",
            "Question Answering", "Dialogue Systems", "Sentiment Analysis",
        ],
    },
    "Computer Vision": {
        "color": "#27AE60",
        "patterns": [
            "object detection", "image segmentation", "image classif",
            "face recognit", "depth estimation", "3d detection", "point cloud",
            "action recognit", "pose estimation", "semantic segmentation",
            "instance segmentation", "visual tracking", "scene understand",
            "video classif", "optical flow", "lane detection", "pedestrian",
            "bounding box", "anchor", "feature pyramid", "backbone",
        ],
        "topics": [
            "Object Detection", "Image Segmentation",
            "Video Understanding", "3D Vision",
            "Face Recognition", "Pose Estimation",
        ],
    },
    "Generative Models": {
        "color": "#8E44AD",
        "patterns": [
            "diffusion model", "generative adversarial", " gan", "image generation",
            "text-to-image", "image synthesis", "stable diffusion",
            "variational autoencoder", " vae", "flow model", "score-based",
            "denoising", "image editing", "style transfer", "super resolution",
            "inpainting", "image-to-image", "video generation", "latent",
            "noise prediction", "ddpm", "score matching",
        ],
        "topics": [
            "Diffusion Models", "Generative Adversarial Networks",
            "Image Synthesis", "Text-to-Image Generation",
            "Video Generation", "Variational Methods",
        ],
    },
    "Reinforcement Learning": {
        "color": "#D35400",
        "patterns": [
            "reinforcement learning", "policy gradient", "reward",
            "q-learning", "multi-agent", "policy optim", "actor critic",
            "model-based rl", "offline rl", "rl agent", "markov",
            "exploration", "value function", "policy network", "curriculum",
            "imitation learning", "inverse reinforcement",
        ],
        "topics": [
            "Policy Optimization", "Multi-Agent RL",
            "Model-Based RL", "Offline RL",
            "Reward Shaping", "Imitation Learning",
        ],
    },
    "Graph & Knowledge": {
        "color": "#16A085",
        "patterns": [
            "graph neural", "knowledge graph", "graph convolution",
            "node classif", "link prediction", "graph attention",
            "graph representation", "knowledge base", "relational",
            "graph transform", "heterogeneous graph", "graph embedding",
            "graph matching", "scene graph", "entity",
        ],
        "topics": [
            "Graph Neural Networks", "Knowledge Graphs",
            "Node Classification", "Link Prediction",
            "Relational Reasoning", "Graph Transformers",
        ],
    },
    "Self-supervised & Few-shot": {
        "color": "#2980B9",
        "patterns": [
            "contrastive learning", "self-supervised", "self supervis",
            "masked", "few-shot", "zero-shot", "meta-learning",
            "transfer learning", "domain adaptation", "domain generali",
            "clip", "representation learning", "unsupervised",
            "semi-supervised", "label efficient", "data augment",
        ],
        "topics": [
            "Contrastive Learning", "Self-supervised Pretraining",
            "Few-shot Learning", "Meta-Learning",
            "Domain Adaptation", "Transfer Learning",
        ],
    },
    "Optimization & Theory": {
        "color": "#F39C12",
        "patterns": [
            "gradient descent", "convergence", "stochastic optim",
            "generalization", "regret", "optimization algorithm",
            "neural tangent", "loss landscape", "federated",
            "differential privacy", "robustness", "adversarial",
            "certified", "bound", "pac learning", "sample complexity",
            "attention mechanism", "normalization",
        ],
        "topics": [
            "Gradient Optimization", "Generalization Theory",
            "Federated Learning", "Adversarial Robustness",
            "Privacy-Preserving ML", "Attention Mechanisms",
        ],
    },
    "Multimodal Learning": {
        "color": "#E91E8C",
        "patterns": [
            "multimodal", "vision language", "visual question",
            "cross-modal", "image-text", "audio visual",
            "vision-language", "visual grounding", "image caption",
            "visual reasoning", "vqa", "image text", "text image",
            "speech recognit", "audio", "speech synthesis",
        ],
        "topics": [
            "Vision-Language Models", "Visual Question Answering",
            "Image Captioning", "Audio-Visual Learning",
            "Speech Recognition", "Cross-modal Retrieval",
        ],
    },
}

STOPWORDS_EXTRA = {
    "based", "model", "models", "method", "methods", "approach", "propose",
    "proposed", "paper", "using", "show", "learning", "data", "network",
    "networks", "deep", "neural", "training", "performance", "state", "art",
    "existing", "new", "large", "result", "results", "novel", "framework",
    "work", "task", "tasks", "ability", "achieve", "achieves", "improve",
    "improvement", "high", "low", "set", "use", "used", "also", "two",
    "different", "various", "across", "multiple", "recent",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_keywords(kw_series):
    def _parse(s):
        if pd.isna(s) or not str(s).strip():
            return []
        return [k.strip().lower() for k in re.split(r"[;,]", str(s)) if k.strip()]
    return kw_series.fillna("").apply(_parse)


def assign_subfield(kw_list, threshold=1):
    """Return list of subfields this paper belongs to (may overlap)."""
    text = " ".join(kw_list).lower()
    matches = []
    for sf, info in SUBFIELDS.items():
        hits = sum(1 for p in info["patterns"] if p in text)
        if hits >= threshold:
            matches.append(sf)
    return matches if matches else ["Other"]


def paper_primary_subfield(kw_list):
    """Return the single best-matching subfield."""
    text = " ".join(kw_list).lower()
    best, best_hits = "Other", 0
    for sf, info in SUBFIELDS.items():
        hits = sum(1 for p in info["patterns"] if p in text)
        if hits > best_hits:
            best, best_hits = sf, hits
    return best


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
main = df[df["paper_type"].astype(str).str.strip().str.lower() == "main"].copy()
main = main[main["keywords"].notna() & (main["keywords"].astype(str).str.strip() != "")].copy()
main = main[main["keywords"].astype(str).str.strip() != "nan"].copy()
# Remove remaining Topic: artifacts
main = main[~main["keywords"].astype(str).str.strip().str.startswith("Topic:")].copy()
main["_kw_list"] = parse_keywords(main["keywords"])
main["_subfield"] = main["_kw_list"].apply(paper_primary_subfield)
print(f"  Main-conference papers with keywords: {len(main)}")
print(f"  Years: {sorted(main['year'].dropna().unique().tolist())}")
print(f"  Conferences: {sorted(main['conference'].dropna().unique().tolist())}")

# Subfield paper counts
sf_counts = main["_subfield"].value_counts()
print("\nPapers per subfield:")
for sf, cnt in sf_counts.items():
    print(f"  {sf:<35} {cnt:>5}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Word cloud grid (one per subfield)
# ═══════════════════════════════════════════════════════════════════════════════
if HAS_WORDCLOUD:
    print("\nGenerating word cloud grid …")
    sf_list = list(SUBFIELDS.keys())
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    fig.patch.set_facecolor("white")

    for idx, sf in enumerate(sf_list):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("white")

        # Collect keyword words for this subfield
        sub = main[main["_subfield"] == sf]
        word_freq = Counter()
        for kws in sub["_kw_list"]:
            for phrase in kws:
                for word in phrase.split():
                    word = re.sub(r"[^a-z]", "", word.lower())
                    if word and word not in STOPWORDS_EXTRA and len(word) > 2:
                        word_freq[word] += 1

        color = SUBFIELDS[sf]["color"]

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(color)
            # Slightly vary brightness
            factor = 0.6 + 0.4 * (font_size / 80)
            r = min(1.0, rgb[0] * factor + 0.2 * (1 - factor))
            g = min(1.0, rgb[1] * factor + 0.2 * (1 - factor))
            b = min(1.0, rgb[2] * factor + 0.2 * (1 - factor))
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        if word_freq:
            wc = WordCloud(
                width=500, height=300, background_color="white",
                max_words=80, prefer_horizontal=0.85,
                collocations=False, color_func=color_func,
            ).generate_from_frequencies(word_freq)
            ax.imshow(wc, interpolation="bilinear")

        ax.set_title(sf, fontsize=11, fontweight="bold", color=color, pad=6)
        ax.axis("off")
        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    fig.suptitle("Research Topic Word Clouds by Subfield (Main Conference Papers, 2017–2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "figA_wordcloud_grid.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {out}")
else:
    print("  Skipping word clouds (wordcloud not installed).")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Subfield mind map / hierarchy tree
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating subfield mind map …")

sf_list = list(SUBFIELDS.keys())
n_sf = len(sf_list)

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# Root node
ROOT_X, ROOT_Y = 11, 7
root_box = FancyBboxPatch((ROOT_X - 1.5, ROOT_Y - 0.55), 3, 1.1,
                           boxstyle="round,pad=0.15", linewidth=2,
                           edgecolor="#2C3E50", facecolor="#2C3E50")
ax.add_patch(root_box)
ax.text(ROOT_X, ROOT_Y, "AI Research\n(2017-2025)", ha="center", va="center",
        fontsize=22, fontweight="bold", color="white")

# Subfield positions: 4 left, 4 right
left_sfs  = sf_list[:4]
right_sfs = sf_list[4:]
left_ys   = np.linspace(11.5, 2.5, 4)
right_ys  = np.linspace(11.5, 2.5, 4)
LEFT_X, RIGHT_X = 3.8, 18.2
TOPIC_LEFT_X, TOPIC_RIGHT_X = 1.0, 21.0

for i, (sf, ys) in enumerate(zip(left_sfs, left_ys)):
    color = SUBFIELDS[sf]["color"]
    topics = SUBFIELDS[sf]["topics"]
    cnt = sf_counts.get(sf, 0)

    # Line root -> subfield
    ax.plot([ROOT_X - 1.5, LEFT_X + 1.5], [ROOT_Y, ys],
            color=color, linewidth=1.5, alpha=0.6, zorder=1)

    # Subfield box
    box = FancyBboxPatch((LEFT_X - 1.5, ys - 0.45), 3, 0.9,
                          boxstyle="round,pad=0.1", linewidth=1.8,
                          edgecolor=color, facecolor=color + "22")
    ax.add_patch(box)
    ax.text(LEFT_X, ys, f"{sf}\n({cnt} papers)", ha="center", va="center",
            fontsize=15, fontweight="bold", color=color)

    # Topic bullets
    n_topics = len(topics)
    topic_ys = np.linspace(ys + (n_topics - 1) * 0.28, ys - (n_topics - 1) * 0.28, n_topics)
    for j, (tp, ty) in enumerate(zip(topics, topic_ys)):
        ax.plot([LEFT_X - 1.5, TOPIC_LEFT_X + 0.05], [ys, ty],
                color=color, linewidth=0.8, alpha=0.4, linestyle="--")
        ax.text(TOPIC_LEFT_X, ty, f"• {tp}", ha="left", va="center",
                fontsize=13, color="#444444")

for i, (sf, ys) in enumerate(zip(right_sfs, right_ys)):
    color = SUBFIELDS[sf]["color"]
    topics = SUBFIELDS[sf]["topics"]
    cnt = sf_counts.get(sf, 0)

    ax.plot([ROOT_X + 1.5, RIGHT_X - 1.5], [ROOT_Y, ys],
            color=color, linewidth=1.5, alpha=0.6, zorder=1)

    box = FancyBboxPatch((RIGHT_X - 1.5, ys - 0.45), 3, 0.9,
                          boxstyle="round,pad=0.1", linewidth=1.8,
                          edgecolor=color, facecolor=color + "22")
    ax.add_patch(box)
    ax.text(RIGHT_X, ys, f"{sf}\n({cnt} papers)", ha="center", va="center",
            fontsize=15, fontweight="bold", color=color)

    n_topics = len(topics)
    topic_ys = np.linspace(ys + (n_topics - 1) * 0.28, ys - (n_topics - 1) * 0.28, n_topics)
    for j, (tp, ty) in enumerate(zip(topics, topic_ys)):
        ax.plot([RIGHT_X + 1.5, TOPIC_RIGHT_X - 0.05], [ys, ty],
                color=color, linewidth=0.8, alpha=0.4, linestyle="--")
        ax.text(TOPIC_RIGHT_X, ty, f"{tp} •", ha="right", va="center",
                fontsize=13, color="#444444")

ax.set_title("Subfield Division of AI Research (ML Conferences 2017-2025)",
             fontsize=22, fontweight="bold", pad=12)
plt.tight_layout()
out = OUTPUT_DIR / "figB_subfield_mindmap.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
plt.close()
print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Topic distribution over time (stacked area + line chart)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating topic distribution over time …")

years = sorted([y for y in main["year"].dropna().unique() if 2017 <= y <= 2025])
sf_names = list(SUBFIELDS.keys())
colors   = [SUBFIELDS[sf]["color"] for sf in sf_names]

# Build year × subfield matrix (proportion of papers)
matrix = pd.DataFrame(index=years, columns=sf_names, dtype=float)
for y in years:
    yr_df = main[main["year"] == y]
    total = len(yr_df)
    for sf in sf_names:
        cnt = (yr_df["_subfield"] == sf).sum()
        matrix.loc[y, sf] = cnt / total if total > 0 else 0

matrix = matrix.fillna(0)

# Smooth with rolling average for area chart aesthetics
matrix_smooth = matrix.copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("white")

# ── Left: stacked area chart ──
ax1.set_facecolor("white")
matrix_smooth_vals = matrix_smooth.values.T  # shape: (n_subfields, n_years)
ax1.stackplot(years, matrix_smooth_vals, labels=sf_names,
              colors=colors, alpha=0.85)
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Proportion of Papers", fontsize=11)
ax1.set_title("Subfield Distribution over Time\n(Stacked Area)", fontsize=12, fontweight="bold")
ax1.set_xticks(years)
ax1.set_xticklabels([str(y) for y in years], rotation=45)
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax1.legend(loc="upper left", fontsize=7, framealpha=0.8,
           bbox_to_anchor=(0.0, 1.0), ncol=1)
ax1.grid(axis="y", alpha=0.3)
sns.despine(ax=ax1)

# ── Right: line chart ──
ax2.set_facecolor("white")
for sf, color in zip(sf_names, colors):
    vals = matrix[sf].values.astype(float)
    ax2.plot(years, vals, marker="o", label=sf, color=color,
             linewidth=2, markersize=5)

ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Proportion of Papers", fontsize=11)
ax2.set_title("Subfield Trends over Time\n(Line Chart)", fontsize=12, fontweight="bold")
ax2.set_xticks(years)
ax2.set_xticklabels([str(y) for y in years], rotation=45)
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax2.legend(loc="upper left", fontsize=7, framealpha=0.8,
           bbox_to_anchor=(0.0, 1.0), ncol=1)
ax2.grid(alpha=0.3)
sns.despine(ax=ax2)

fig.suptitle("Topic Distribution over Time — ML Conferences (2017–2025)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = OUTPUT_DIR / "figC_topic_distribution_time.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
plt.close()
print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — Top keywords per subfield (horizontal bar, like heatmap supplement)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating top-keywords-per-subfield bar chart …")

ncols, nrows = 4, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
fig.patch.set_facecolor("white")

for idx, sf in enumerate(sf_list):
    ax = axes[idx // ncols][idx % ncols]
    sub = main[main["_subfield"] == sf]
    phrase_counter = Counter()
    for kws in sub["_kw_list"]:
        for phrase in kws:
            if len(phrase.split()) >= 2:   # keep multi-word phrases
                phrase_counter[phrase] += 1

    top = phrase_counter.most_common(15)
    if not top:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(sf, fontsize=9)
        continue

    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]
    color  = SUBFIELDS[sf]["color"]

    bars = ax.barh(range(len(labels)), values, color=color, alpha=0.75, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Frequency", fontsize=8)
    ax.set_title(sf, fontsize=9, fontweight="bold", color=color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Top Keyphrases per Subfield — ML Conferences (2017–2025)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=1.5)
out = OUTPUT_DIR / "figD_top_keyphrases_per_subfield.pdf"
fig.savefig(out, bbox_inches="tight", dpi=200)
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
plt.close()
print(f"  Saved -> {out}")

# ── Print paper-count matrix for reference ────────────────────────────────────
print("\n--- Paper count matrix (subfield x year) ---")
count_matrix = pd.DataFrame(index=years, columns=sf_names, dtype=int)
for y in years:
    yr_df = main[main["year"] == y]
    for sf in sf_names:
        count_matrix.loc[y, sf] = (yr_df["_subfield"] == sf).sum()
print(count_matrix.to_string())

print(f"\nAll figures saved to: {OUTPUT_DIR}/")
