"""
Topic analysis on main-conference papers only (2017–2025, ACL, CVPR, ICLR, ICML, NeurIPS).
Reads the Excel (e.g. merged_papers_keywords_*.xlsx), filters to paper_type == 'main',
and produces: papers per year, top topics, heatmaps, line charts, and review-paper figures
(growth, LLM revolution, diffusion, RL, consolidation, word clouds).
Optional: pip install wordcloud for topic word clouds.
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# INPUT_EXCEL = SCRIPT_DIR / "ml_conferences_accepted_papers_2017_2025_v6.xlsx"
INPUT_EXCEL = SCRIPT_DIR / "merged_papers_keywords_keybert.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "topic_analysis_figures_keybert_v5"
TOP_N_KEYWORDS = 10
HEATMAP_TOP_N = 15
TOP_N_TOPICS_LINE = 25  # top N topics per conference for line chart over years
PAPER_TYPE_MAIN = "main"

# Style
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 120


def normalize_keyword(k: str) -> str:
    """Lowercase and strip; collapse spaces."""
    if not isinstance(k, str):
        return ""
    return " ".join(k.lower().strip().split())


def parse_keywords(keywords_series):
    """Parse 'keywords' column: split by ; or , and normalize. Returns list of keywords per row."""
    def _parse(s):
        if pd.isna(s) or not str(s).strip():
            return []
        s = str(s)
        parts = re.split(r"\s*[;,]\s*", s)
        return [normalize_keyword(p) for p in parts if normalize_keyword(p)]
    return keywords_series.fillna("").apply(_parse)


def load_main_papers(excel_path: Path) -> pd.DataFrame:
    """Load Excel and keep only main-conference papers (flag workshops excluded)."""
    df = pd.read_excel(excel_path, engine="openpyxl")
    if "paper_type" not in df.columns:
        df["paper_type"] = "main"
    main = df[df["paper_type"].astype(str).str.strip().str.lower() == PAPER_TYPE_MAIN].copy()
    main["year"] = pd.to_numeric(main["year"], errors="coerce").astype("Int64")
    return main


def papers_per_year_per_conference(df: pd.DataFrame) -> pd.DataFrame:
    """Count of papers by year and conference."""
    return df.groupby(["year", "conference"]).size().reset_index(name="count")


def get_keyword_counts(df: pd.DataFrame):
    """Flatten keywords and count. Returns Series (keyword -> count)."""
    lists = parse_keywords(df["keywords"])
    all_kw = []
    for L in lists:
        all_kw.extend(L)
    return pd.Series(all_kw).value_counts()


def get_top_keywords_per_conference(df: pd.DataFrame, top_n: int = TOP_N_KEYWORDS):
    """For each conference, return top N keywords as a dict conference -> list of (kw, count)."""
    out = {}
    for conf in df["conference"].dropna().unique():
        sub = df[df["conference"] == conf]
        counts = get_keyword_counts(sub)
        out[str(conf)] = list(counts.head(top_n).items())
    return out


def keyword_counts_by_year(df: pd.DataFrame, top_n: int = HEATMAP_TOP_N):
    """DataFrame: rows = top N keywords overall, cols = years, values = count in that year."""
    overall = get_keyword_counts(df)
    top_keywords = overall.head(top_n).index.tolist()
    lists = parse_keywords(df["keywords"])
    df = df.copy()
    df["_kw_list"] = lists
    year_counts = {y: {} for y in df["year"].dropna().unique()}
    for y in year_counts:
        sub = df[df["year"] == y]
        flat = []
        for L in sub["_kw_list"]:
            flat.extend(L)
        for kw in top_keywords:
            year_counts[y][kw] = flat.count(kw)
    return pd.DataFrame(year_counts).reindex(top_keywords).fillna(0).astype(int)


def get_topic_count_over_years(df: pd.DataFrame, topic_patterns: list, normalize_key=True):
    """For keywords matching any of topic_patterns (substring, case-insensitive), return Series year -> count."""
    lists = parse_keywords(df["keywords"])
    df = df.copy()
    df["_kw_list"] = lists
    years = sorted(df["year"].dropna().unique())
    out = {}
    for y in years:
        sub = df[df["year"] == y]
        flat = []
        for L in sub["_kw_list"]:
            flat.extend(L)
        count = 0
        for kw in flat:
            k = kw.lower() if normalize_key else kw
            if any(p.lower() in k for p in topic_patterns):
                count += 1
        out[y] = count
    return pd.Series(out)


def plot_papers_per_year_line(df_main: pd.DataFrame):
    """Line chart: papers per year, one line per conference."""
    ppt = papers_per_year_per_conference(df_main)
    pivot = ppt.pivot(index="year", columns="conference", values="count").fillna(0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=col, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Main conference: Papers accepted per year by conference")
    ax.legend(loc="best")
    ax.set_xticks(pivot.index)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_papers_per_year_by_conference.png", bbox_inches="tight")
    plt.close()


def plot_papers_per_year_stacked_bar(df_main: pd.DataFrame):
    """Stacked bar: year x count, stacked by conference."""
    ppt = papers_per_year_per_conference(df_main)
    pivot = ppt.pivot(index="year", columns="conference", values="count").fillna(0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Main conference: Papers per year by conference (stacked)")
    ax.legend(title="Conference", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_papers_per_year_stacked_bar.png", bbox_inches="tight")
    plt.close()


def plot_pie_papers_by_conference(df_main: pd.DataFrame):
    """Pie chart: share of papers by conference (main only)."""
    counts = df_main["conference"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Main conference: Share of papers by conference")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_pie_papers_by_conference.png", bbox_inches="tight")
    plt.close()


def plot_top10_overall(df_main: pd.DataFrame):
    """Bar chart: top 10 topics (keywords) across all main papers."""
    counts = get_keyword_counts(df_main).head(TOP_N_KEYWORDS)
    fig, ax = plt.subplots(figsize=(9, 6))
    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="gray")
    ax.set_xlabel("Number of papers")
    ax.set_ylabel("Topic (keyword)")
    ax.set_title(f"Main conference: Top {TOP_N_KEYWORDS} topics (all conferences)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_top10_topics_overall.png", bbox_inches="tight")
    plt.close()


def plot_top10_per_conference(df_main: pd.DataFrame):
    """One horizontal bar chart per conference: top 10 topics."""
    top_per_conf = get_top_keywords_per_conference(df_main, top_n=TOP_N_KEYWORDS)
    conferences = sorted(top_per_conf.keys())
    n = len(conferences)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow))
    if nrow == 1 and ncol == 1:
        axes = [[axes]]
    elif nrow == 1:
        axes = [axes]
    for idx, conf in enumerate(conferences):
        r, c = idx // ncol, idx % ncol
        ax = axes[r][c]
        items = top_per_conf[conf]
        if not items:
            ax.text(0.5, 0.5, "No keywords", ha="center", va="center")
            ax.set_title(conf)
            continue
        kws = [x[0] for x in items]
        vals = [x[1] for x in items]
        ax.barh(range(len(kws)), vals, color="steelblue", edgecolor="gray")
        ax.set_yticks(range(len(kws)))
        ax.set_yticklabels(kws, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(f"{conf}: Top {TOP_N_KEYWORDS} topics")
    for idx in range(len(conferences), nrow * ncol):
        r, c = idx // ncol, idx % ncol
        axes[r][c].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_top10_topics_per_conference.png", bbox_inches="tight")
    plt.close()


def plot_heatmap_topics_over_years(df_main: pd.DataFrame):
    """Heatmap: top topics (rows) x years (columns), cell = paper count."""
    mat = keyword_counts_by_year(df_main, top_n=HEATMAP_TOP_N)
    if mat.empty:
        return
    mat = mat.reindex(columns=sorted(mat.columns))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(mat, annot=False, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Number of papers"})
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic (keyword)")
    ax.set_title("Main conference: Topic prominence over years (all conferences)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_heatmap_topics_over_years.png", bbox_inches="tight")
    plt.close()


def plot_line_top5_topics_over_years(df_main: pd.DataFrame):
    """Line chart: trend of top 5 topics over years (all conferences)."""
    counts = get_keyword_counts(df_main)
    top5 = counts.head(5).index.tolist()
    lists = parse_keywords(df_main["keywords"])
    df_main = df_main.copy()
    df_main["_kw_list"] = lists
    years = sorted(df_main["year"].dropna().unique())
    data = {kw: [] for kw in top5}
    for y in years:
        sub = df_main[df_main["year"] == y]
        flat = []
        for L in sub["_kw_list"]:
            flat.extend(L)
        for kw in top5:
            data[kw].append(flat.count(kw))
    fig, ax = plt.subplots(figsize=(10, 5))
    for kw in top5:
        ax.plot(years, data[kw], marker="o", label=kw, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Main conference: Top 5 topics over years (all conferences)")
    ax.legend(loc="best")
    ax.set_xticks(years)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_line_top5_topics_over_years.png", bbox_inches="tight")
    plt.close()


def plot_heatmap_per_conference(df_main: pd.DataFrame):
    """One heatmap per conference: topics x years."""
    for conf in sorted(df_main["conference"].dropna().unique()):
        sub = df_main[df_main["conference"] == conf]
        mat = keyword_counts_by_year(sub, top_n=min(HEATMAP_TOP_N, 12))
        if mat.empty:
            continue
        mat = mat.reindex(columns=sorted(mat.columns))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mat, annot=False, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Count"})
        ax.set_xlabel("Year")
        ax.set_ylabel("Topic")
        ax.set_title(f"Main conference: Topic prominence over years — {conf}")
        plt.xticks(rotation=0)
        plt.tight_layout()
        safe_name = re.sub(r"[^\w\-]", "_", str(conf))
        plt.savefig(OUTPUT_DIR / f"08_heatmap_topics_years_{safe_name}.png", bbox_inches="tight")
        plt.close()


def _keyword_counts_by_year_for_conference(df_sub: pd.DataFrame, top_keywords: list) -> pd.DataFrame:
    """For a conference subset, return DataFrame rows=keywords, cols=years, values=count."""
    lists = parse_keywords(df_sub["keywords"])
    df_sub = df_sub.copy()
    df_sub["_kw_list"] = lists
    years = sorted(df_sub["year"].dropna().unique())
    year_counts = {y: {} for y in years}
    for y in years:
        sub = df_sub[df_sub["year"] == y]
        flat = []
        for L in sub["_kw_list"]:
            flat.extend(L)
        for kw in top_keywords:
            year_counts[y][kw] = flat.count(kw)
    return pd.DataFrame(year_counts).reindex(top_keywords).fillna(0).astype(int)


def plot_top25_topics_per_conference_over_years(df_main: pd.DataFrame):
    """For each conference: line chart of top 25 topics over years (one line per topic)."""
    top_per_conf = get_top_keywords_per_conference(df_main, top_n=TOP_N_TOPICS_LINE)
    # Colors for many lines (tab20 + tab20b = 40 distinct colors)
    n1, n2 = 20, 20
    colors = (
        [plt.cm.tab20(i / max(n1 - 1, 1)) for i in range(n1)]
        + [plt.cm.tab20b(i / max(n2 - 1, 1)) for i in range(n2)]
    )

    for conf in sorted(top_per_conf.keys()):
        items = top_per_conf[conf]
        if not items:
            continue
        top_keywords = [x[0] for x in items]
        sub = df_main[df_main["conference"] == conf]
        mat = _keyword_counts_by_year_for_conference(sub, top_keywords)
        if mat.empty:
            continue
        mat = mat.reindex(columns=sorted(mat.columns))
        years = mat.columns.tolist()

        fig, ax = plt.subplots(figsize=(14, 7))
        for idx, kw in enumerate(top_keywords):
            ax.plot(
                years,
                mat.loc[kw].values,
                marker="o",
                markersize=3,
                label=kw,
                color=colors[idx % len(colors)],
                linewidth=1.5,
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of papers")
        ax.set_title(f"Main conference: Top {TOP_N_TOPICS_LINE} topics over years — {conf}")
        ax.set_xticks(years)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        safe_name = re.sub(r"[^\w\-]", "_", str(conf))
        plt.savefig(OUTPUT_DIR / f"09_line_top25_topics_years_{safe_name}.png", bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Review-paper figures: exponential growth, LLM revolution, diffusion, RL, etc.
# ---------------------------------------------------------------------------

def plot_exponential_growth_total_and_stacked(df_main: pd.DataFrame):
    """Total papers per year (line) + stacked area by conference. Highlights exponential growth."""
    ppt = papers_per_year_per_conference(df_main)
    pivot = ppt.pivot(index="year", columns="conference", values="count").fillna(0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.sort_index()
    total = pivot.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1.fill_between(total.index, total.values, alpha=0.3, color="steelblue")
    ax1.plot(total.index, total.values, marker="o", linewidth=2.5, color="steelblue", markersize=8)
    ax1.set_ylabel("Total papers")
    ax1.set_title("Exponential growth: total accepted papers across five conferences (ACL, CVPR, ICLR, ICML, NeurIPS)")
    ax1.set_xticks(total.index)
    ax1.grid(True, alpha=0.3)

    pivot.plot(kind="area", stacked=True, ax=ax2, alpha=0.85, colormap="Set3")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Papers")
    ax2.set_title("Stacked volume by conference")
    ax2.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_exponential_growth_total_and_stacked.png", bbox_inches="tight")
    plt.close()


def plot_2017_vs_2025_comparison(df_main: pd.DataFrame):
    """Side-by-side: papers per conference in 2017 vs 2025. Emphasizes growth."""
    ppt = papers_per_year_per_conference(df_main)
    y2017 = ppt[ppt["year"] == 2017].set_index("conference")["count"]
    y2025 = ppt[ppt["year"] == 2025].set_index("conference")["count"]
    confs = sorted(set(y2017.index) | set(y2025.index))
    x = range(len(confs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w for i in x], [y2017.get(c, 0) for c in confs], width=w, label="2017", color="cornflowerblue")
    ax.bar([i + w for i in x], [y2025.get(c, 0) for c in confs], width=w, label="2025", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(confs)
    ax.set_ylabel("Number of papers")
    ax.set_title("Research volume: 2017 vs 2025 by conference")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_2017_vs_2025_by_conference.png", bbox_inches="tight")
    plt.close()


def plot_llm_revolution_over_years(df_main: pd.DataFrame):
    """Line chart: LLM-related topic count over years (post-2023 surge)."""
    patterns = ["large language model", "llm", "language model"]
    llm = get_topic_count_over_years(df_main, patterns)
    years = sorted(llm.index)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years, llm.values, alpha=0.4, color="green")
    ax.plot(years, llm.values, marker="o", linewidth=2.5, color="darkgreen", markersize=8)
    ax.axvspan(2023, 2026, alpha=0.15, color="green", label="Post-2023")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers (LLM-related)")
    ax.set_title('The "LLM Revolution": papers with large language model / LLM topic (2017–2025)')
    ax.set_xticks(years)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_llm_revolution_over_years.png", bbox_inches="tight")
    plt.close()


def plot_acl_llm_vs_traditional(df_main: pd.DataFrame):
    """ACL: LLM vs neural machine translation vs named entity recognition (e.g. 2025 or latest)."""
    acl = df_main[df_main["conference"] == "ACL"]
    if acl.empty:
        return
    counts = get_keyword_counts(acl)
    # Match common topic names (normalized)
    by_label = {}
    for kw, c in counts.items():
        k = kw.lower()
        if "large language" in k or k == "llm":
            by_label["Large language models"] = by_label.get("Large language models", 0) + c
        elif "neural machine translation" in k or "machine translation" in k:
            by_label["Neural machine translation"] = by_label.get("Neural machine translation", 0) + c
        elif "named entity" in k or "ner" in k:
            by_label["Named entity recognition"] = by_label.get("Named entity recognition", 0) + c
    if not by_label:
        return
    labels = list(by_label.keys())
    vals = [by_label[l] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    ax.barh(labels, vals, color=colors[: len(labels)])
    ax.set_xlabel("Number of papers")
    ax.set_title("ACL: LLM vs traditional NLP topics (all years combined)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_acl_llm_vs_traditional.png", bbox_inches="tight")
    plt.close()


def plot_diffusion_rise_over_years(df_main: pd.DataFrame):
    """Line chart: diffusion models topic over years (meteoric rise from ~2022)."""
    patterns = ["diffusion"]
    diff = get_topic_count_over_years(df_main, patterns)
    years = sorted(diff.index)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, diff.values, marker="s", linewidth=2.5, color="purple", markersize=8)
    ax.fill_between(years, diff.values, alpha=0.3, color="purple")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Rapid emergence of diffusion models (2017–2025)")
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_diffusion_rise_over_years.png", bbox_inches="tight")
    plt.close()


def plot_rl_pillar_iclr_neurips(df_main: pd.DataFrame):
    """Reinforcement learning as top topic at ICLR and NeurIPS: count over years."""
    patterns = ["reinforcement learning", "rl "]
    rl_all = get_topic_count_over_years(df_main, patterns)
    for conf in ["ICLR", "NeurIPS"]:
        sub = df_main[df_main["conference"] == conf]
        if sub.empty:
            continue
        rl = get_topic_count_over_years(sub, patterns)
        years = sorted(rl.index)
        plt.figure(figsize=(9, 4))
        plt.plot(years, rl.values, marker="o", linewidth=2, label=conf, color="coral" if conf == "ICLR" else "teal")
        plt.xlabel("Year")
        plt.ylabel("Number of papers")
        plt.title(f"Reinforcement learning as a foundational pillar — {conf}")
        plt.xticks(years)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = re.sub(r"[^\w\-]", "_", conf)
        plt.savefig(OUTPUT_DIR / f"15_rl_over_years_{safe}.png", bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    years = sorted(rl_all.index)
    ax.plot(years, rl_all.values, marker="o", linewidth=2.5, color="darkred", markersize=8)
    ax.fill_between(years, rl_all.values, alpha=0.25, color="darkred")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Reinforcement learning across all five conferences (steady growth)")
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_rl_over_years_all.png", bbox_inches="tight")
    plt.close()


def plot_cvpr_vision_language_llm(df_main: pd.DataFrame):
    """CVPR top 10 with vision-language and LLM highlighted (cross-domain influence)."""
    cvpr = df_main[df_main["conference"] == "CVPR"]
    if cvpr.empty:
        return
    top = get_keyword_counts(cvpr).head(12)
    highlight = []
    for kw in top.index:
        k = kw.lower()
        if "vision" in k and "language" in k:
            highlight.append(kw)
        if "large language" in k or k == "llm":
            highlight.append(kw)
    colors = ["#e74c3c" if kw in highlight else "steelblue" for kw in top.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax, color=colors, edgecolor="gray")
    ax.set_xlabel("Number of papers")
    ax.set_title("CVPR top topics: cross-domain influence (vision-language & LLM highlighted)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "16_cvpr_top_with_vision_language_llm.png", bbox_inches="tight")
    plt.close()


def plot_consolidation_acl_top5_share(df_main: pd.DataFrame):
    """ACL: share of top 5 topics over years (narrowing focus)."""
    acl = df_main[df_main["conference"] == "ACL"]
    if acl.empty:
        return
    counts = get_keyword_counts(acl)
    top5 = counts.head(5).index.tolist()
    lists = parse_keywords(acl["keywords"])
    acl = acl.copy()
    acl["_kw_list"] = lists
    years = sorted(acl["year"].dropna().unique())
    total_papers = acl.groupby("year").size()
    top5_count = {}
    for y in years:
        sub = acl[acl["year"] == y]
        flat = []
        for L in sub["_kw_list"]:
            flat.extend(L)
        top5_count[y] = sum(flat.count(k) for k in top5)
    share = pd.Series(top5_count) / total_papers.reindex(years).fillna(1)
    share = share.fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years, share.values * 100, marker="o", linewidth=2, color="darkblue")
    ax.fill_between(years, share.values * 100, alpha=0.2, color="darkblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (%)")
    ax.set_title("ACL: consolidation — share of papers with top-5 topics (narrowing focus)")
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_acl_consolidation_top5_share.png", bbox_inches="tight")
    plt.close()


def plot_wordclouds(df_main: pd.DataFrame):
    """Word clouds: overall and per conference (optional dependency: wordcloud)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return
    lists = parse_keywords(df_main["keywords"])
    freq = {}
    for L in lists:
        for w in L:
            w = w.strip()
            if len(w) > 1:
                freq[w] = freq.get(w, 0) + 1
    if not freq:
        return
    wc = WordCloud(width=1200, height=600, background_color="white", max_words=80, colormap="viridis").generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Topic word cloud: all conferences (2017–2025)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "18_wordcloud_overall.png", bbox_inches="tight")
    plt.close()

    for conf in sorted(df_main["conference"].dropna().unique()):
        sub = df_main[df_main["conference"] == conf]
        lists = parse_keywords(sub["keywords"])
        freq = {}
        for L in lists:
            for w in L:
                w = w.strip()
                if len(w) > 1:
                    freq[w] = freq.get(w, 0) + 1
        if not freq:
            continue
        wc = WordCloud(width=1000, height=500, background_color="white", max_words=50, colormap="Set3").generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic word cloud — {conf}")
        plt.tight_layout()
        safe = re.sub(r"[^\w\-]", "_", str(conf))
        plt.savefig(OUTPUT_DIR / f"18_wordcloud_{safe}.png", bbox_inches="tight")
        plt.close()


def plot_neurips_share_pie(df_main: pd.DataFrame):
    """Pie: NeurIPS share of total papers (e.g. 2025) — '28% of total'."""
    ppt = papers_per_year_per_conference(df_main)
    y2025 = ppt[ppt["year"] == 2025]
    total = y2025["count"].sum()
    if total == 0:
        return
    neurips = y2025[y2025["conference"] == "NeurIPS"]["count"].sum()
    others = total - neurips
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie([neurips, others], labels=[f"NeurIPS ({neurips:,})", f"Others ({others:,})"], autopct="%1.1f%%", startangle=90, colors=["#e74c3c", "#bdc3c7"])
    ax.set_title("2025: NeurIPS share of papers across five conferences")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "19_neurips_share_2025.png", bbox_inches="tight")
    plt.close()


def plot_emerging_topics_timeline(df_main: pd.DataFrame):
    """Small multiples or bars: emerging topics (e.g. RAG, 3d gaussian) over years."""
    patterns_list = [
        ("Retrieval augmented generation / RAG", ["retrieval augmented", "rag ", " retrieval augment"]),
        ("3D Gaussian splatting", ["3d gaussian", "gaussian splatting"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (label, patterns) in zip(axes, patterns_list):
        s = get_topic_count_over_years(df_main, patterns)
        s = s.reindex(sorted(df_main["year"].dropna().unique())).fillna(0)
        ax.bar(s.index, s.values, color="teal", edgecolor="white")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of papers")
        ax.set_title(label)
        ax.set_xticks(s.index)
    plt.suptitle("Emerging specialized topics (2017–2025)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "20_emerging_topics_timeline.png", bbox_inches="tight")
    plt.close()


def write_summary_tables(df_main: pd.DataFrame):
    """Write CSV summary tables for papers-per-year and top topics."""
    ppt = papers_per_year_per_conference(df_main)
    ppt.to_csv(OUTPUT_DIR / "summary_papers_per_year_per_conference.csv", index=False)
    overall = get_keyword_counts(df_main).head(TOP_N_KEYWORDS)
    overall.to_csv(OUTPUT_DIR / "summary_top10_topics_overall.csv", header=["count"])
    top_per_conf = get_top_keywords_per_conference(df_main, top_n=TOP_N_KEYWORDS)
    rows = []
    for conf, items in top_per_conf.items():
        for kw, cnt in items:
            rows.append({"conference": conf, "keyword": kw, "count": cnt})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "summary_top10_topics_per_conference.csv", index=False)


def main():
    if not INPUT_EXCEL.exists():
        print(f"Input file not found: {INPUT_EXCEL}")
        print("Run fetch_icml_papers.py first to generate the Excel.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data (main conference papers only)...")
    df = load_main_papers(INPUT_EXCEL)
    print(f"  Main papers: {len(df)} (workshops excluded)")

    print("Building summary tables...")
    write_summary_tables(df)

    print("Plots: papers per year...")
    plot_papers_per_year_line(df)
    plot_papers_per_year_stacked_bar(df)

    print("Plots: distribution...")
    plot_pie_papers_by_conference(df)

    print("Plots: top topics...")
    plot_top10_overall(df)
    plot_top10_per_conference(df)

    print("Plots: topics over years...")
    plot_heatmap_topics_over_years(df)
    plot_line_top5_topics_over_years(df)
    plot_heatmap_per_conference(df)
    print("Plots: top 25 topics per conference over years (line charts)...")
    plot_top25_topics_per_conference_over_years(df)

    print("Plots: review-paper figures (growth, LLM, diffusion, RL, consolidation)...")
    plot_exponential_growth_total_and_stacked(df)
    plot_2017_vs_2025_comparison(df)
    plot_llm_revolution_over_years(df)
    plot_acl_llm_vs_traditional(df)
    plot_diffusion_rise_over_years(df)
    plot_rl_pillar_iclr_neurips(df)
    plot_cvpr_vision_language_llm(df)
    plot_consolidation_acl_top5_share(df)
    plot_neurips_share_pie(df)
    plot_emerging_topics_timeline(df)
    print("Plots: word clouds (if wordcloud installed)...")
    plot_wordclouds(df)

    print(f"\nDone. Figures and CSVs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
