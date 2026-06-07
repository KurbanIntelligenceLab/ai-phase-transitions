"""Rebuild all topic-analysis figures from exact dataset values."""
import pandas as pd, numpy as np, re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTDIR = Path('c:/Users/User/Desktop/paper/topic_analysis_figures_keybert_v5')
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150

df = pd.read_excel('c:/Users/User/Desktop/paper/merged_papers_keywords_keybert.xlsx')
df = df[df['paper_type'].astype(str).str.strip().str.lower() == 'main'].copy()
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

def parse_kw(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [k.strip().lower() for k in re.split(r'[;,]', str(s)) if k.strip()]

df['kw_list'] = df['keywords'].apply(parse_kw)
rows = []
for _, row in df.iterrows():
    for kw in row['kw_list']:
        rows.append({'kw': kw, 'year': int(row['year']), 'conf': row['conference']})
kw_df = pd.DataFrame(rows)
years = list(range(2017, 2026))

# ── Fig 05: top-10 per conference (horizontal bar, 5 panels) ------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
confs = ['ACL', 'CVPR', 'ICLR', 'ICML', 'NeurIPS']
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
for ax, conf, color in zip(axes, confs, colors):
    sub = kw_df[kw_df['conf'] == conf]
    top10 = sub['kw'].value_counts().head(10)
    labels = [k[:28] for k in top10.index[::-1]]
    vals = list(top10.values[::-1])
    bars = ax.barh(labels, vals, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=8)
    ax.set_title(conf, fontweight='bold', fontsize=13)
    ax.set_xlabel('Papers', fontsize=10)
    ax.margins(x=0.15)
plt.suptitle('Top-10 Topics per Conference (2017-2025)', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTDIR / '05_top10_topics_per_conference.png', bbox_inches='tight')
plt.close()
print('Saved Fig 05')

# ── Fig 07: top-5 cross-venue line chart --------------------------------------
top5 = kw_df['kw'].value_counts().head(5).index.tolist()
piv = kw_df[kw_df['kw'].isin(top5)].groupby(['kw', 'year']).size().unstack(fill_value=0)
for y in years:
    if y not in piv.columns:
        piv[y] = 0
piv = piv[years]

fig, ax = plt.subplots(figsize=(11, 6))
palette = sns.color_palette('tab10', 5)
for i, kw in enumerate(top5):
    vals = [piv.loc[kw, y] for y in years]
    ax.plot(years, vals, marker='o', label=kw.title()[:30],
            color=palette[i], lw=2)
ax.set_xlabel('Year')
ax.set_ylabel('Papers')
ax.set_title('Top-5 Topics Over Years (All Venues, 2017-2025)', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig(OUTDIR / '07_line_top5_topics_over_years.png', bbox_inches='tight')
plt.close()
print('Saved Fig 07')

# ── Fig 06: heatmap top-15 ----------------------------------------------------
top15 = kw_df['kw'].value_counts().head(15).index.tolist()
piv2 = kw_df[kw_df['kw'].isin(top15)].groupby(['kw', 'year']).size().unstack(fill_value=0)
for y in years:
    if y not in piv2.columns:
        piv2[y] = 0
piv2 = piv2[years]
order = kw_df[kw_df['kw'].isin(top15)]['kw'].value_counts().index
piv2 = piv2.loc[order]

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(piv2, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            linewidths=0.3, cbar_kws={'label': 'Papers'})
ax.set_title('Top-15 Topics Over Years (All Venues)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('')
ax.set_yticklabels(
    [t.get_text().title()[:32] for t in ax.get_yticklabels()], fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / '06_heatmap_topics_over_years.png', bbox_inches='tight')
plt.close()
print('Saved Fig 06')

# ── Fig 12: LLM revolution ----------------------------------------------------
llm_map = {
    'Large Language Models':      'large language models',
    'Large Language Model':       'large language model',
    'LLM':                        'llm',
    'In-Context Learning':        'in-context learning',
    'Retrieval-Augmented Gen.':   'retrieval augmented generation',
}
fig, ax = plt.subplots(figsize=(11, 6))
pal = sns.color_palette('tab10', len(llm_map))
for i, (label, kw) in enumerate(llm_map.items()):
    s = kw_df[kw_df['kw'] == kw].groupby('year').size().reindex(years, fill_value=0)
    ax.plot(years, s.values, marker='o', label=label, color=pal[i], lw=2)
ax.set_title('LLM-Related Topics Over Years (2017-2025)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Papers')
ax.legend(fontsize=9)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig(OUTDIR / '12_llm_revolution_over_years.png', bbox_inches='tight')
plt.close()
print('Saved Fig 12')

# ── Fig 14: Diffusion rise -----------------------------------------------------
diff_map = {
    'Diffusion Models': 'diffusion models',
    'Diffusion Model':  'diffusion model',
    'Flow Matching':    'flow matching',
    'Score Matching':   'score matching',
    'Stable Diffusion': 'stable diffusion',
}
fig, ax = plt.subplots(figsize=(11, 6))
pal2 = sns.color_palette('Blues_d', len(diff_map))
for i, (label, kw) in enumerate(diff_map.items()):
    s = kw_df[kw_df['kw'] == kw].groupby('year').size().reindex(years, fill_value=0)
    ax.plot(years, s.values, marker='o', label=label, color=pal2[i], lw=2)
ax.set_title('Diffusion-Related Topics Over Years (2017-2025)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Papers')
ax.legend(fontsize=9)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig(OUTDIR / '14_diffusion_rise_over_years.png', bbox_inches='tight')
plt.close()
print('Saved Fig 14')

# ── Fig 15: RL three panels ---------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
rl_kw = 'reinforcement learning'
panels = [
    ('All Venues', None,       '#E74C3C'),
    ('ICLR',       'ICLR',    '#3498DB'),
    ('NeurIPS',    'NeurIPS', '#2ECC71'),
]
for ax, (title, conf, color) in zip(axes, panels):
    sub = kw_df if conf is None else kw_df[kw_df['conf'] == conf]
    s = sub[sub['kw'] == rl_kw].groupby('year').size().reindex(years, fill_value=0)
    ax.bar(years, s.values, color=color, alpha=0.8, width=0.7)
    for x, v in zip(years, s.values):
        if v > 0:
            ax.text(x, v + 2, str(v), ha='center', fontsize=7.5)
    ax.set_title('RL - ' + title, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Papers')
    ax.set_xticks(years)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Reinforcement Learning Over Years', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTDIR / '15_rl_over_years_all.png', bbox_inches='tight')
plt.close()
print('Saved Fig 15')

# ── Verification CSV -----------------------------------------------------------
top30 = kw_df['kw'].value_counts().head(30).index.tolist()
piv_v = kw_df[kw_df['kw'].isin(top30)].groupby(['kw', 'year']).size().unstack(fill_value=0)
for y in years:
    if y not in piv_v.columns:
        piv_v[y] = 0
piv_v[years].to_csv(OUTDIR / 'verified_topic_counts_by_year.csv')
print('Saved verification CSV')
print('All figures rebuilt from exact data values.')
