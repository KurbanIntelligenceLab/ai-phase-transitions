"""
Analyze keyword quality across multiple 200-sample draws from existing KeyBERT outputs.
Uses merged_papers_keywords_keybert_filled.xlsx which has KeyBERT keywords already.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter

df = pd.read_excel('merged_papers_keywords_keybert_filled.xlsx', engine='openpyxl')
print(f"Shape: {df.shape}")

# Filter rows that actually have keywords (non-empty, non-Topic:, non-nan)
has_kw = df['keywords'].notna() & \
         (~df['keywords'].astype(str).str.strip().str.startswith('Topic:')) & \
         (df['keywords'].astype(str).str.strip() != '') & \
         (df['keywords'].astype(str).str.strip() != 'nan')
df_kw = df[has_kw].copy()
print(f"Rows with KeyBERT keywords: {len(df_kw)}")

np.random.seed(0)

# ── 3 independent samples of 200 ──────────────────────────────────────────────
for draw in range(1, 4):
    sample = df_kw.sample(200, random_state=draw*7)
    print(f"\n{'='*70}")
    print(f"SAMPLE {draw} of 200 rows")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"[{i:3d}] [{row['conference']} {row['year']}]  {str(row['title'])[:65]}")
        print(f"       {str(row['keywords'])[:130]}")

# ── Aggregate quality metrics across all keywords ─────────────────────────────
print(f"\n{'='*70}")
print("GLOBAL KEYWORD QUALITY METRICS")
print(f"{'='*70}")

all_keywords = []
unigrams = 0
bigrams  = 0
trigrams = 0
total    = 0
generic_words = {
    'model','models','learning','data','method','methods','based','propose',
    'training','show','performance','paper','approach','using','results',
    'state','art','existing','various','different','novel','new','large',
    'framework','system','task','tasks','image','images','network','networks',
    'deep','neural','propose','proposed','demonstrates','achieves','outperforms'
}

for kw_str in df_kw['keywords'].dropna():
    phrases = [k.strip() for k in str(kw_str).split(';') if k.strip()]
    for p in phrases:
        all_keywords.append(p.lower())
        words = p.split()
        total += 1
        if len(words) == 1: unigrams += 1
        elif len(words) == 2: bigrams += 1
        else: trigrams += 1

print(f"Total keyword phrases analyzed: {total:,}")
print(f"  Unigrams : {unigrams:,}  ({unigrams/total*100:.1f}%)")
print(f"  Bigrams  : {bigrams:,}  ({bigrams/total*100:.1f}%)")
print(f"  Trigrams : {trigrams:,}  ({trigrams/total*100:.1f}%)")

kw_counter = Counter(all_keywords)
print(f"\nTop 50 most frequent keyword phrases (ideally should be AI topics):")
for kw, cnt in kw_counter.most_common(50):
    flag = "⚠ GENERIC" if kw in generic_words else ""
    print(f"  {kw:<40} {cnt:>5}x  {flag}")

# ── Check keyword specificity per paper ───────────────────────────────────────
print(f"\n{'='*70}")
print("SPECIFICITY: Papers where keywords are too generic (all words in generic set)")
print(f"{'='*70}")
generic_kw_count = 0
specific_kw_count = 0
sample_generic = []
sample_specific = []

for _, row in df_kw.sample(500, random_state=42).iterrows():
    phrases = [k.strip().lower() for k in str(row['keywords']).split(';') if k.strip()]
    words_all = [w for p in phrases for w in p.split()]
    generic_ratio = sum(1 for w in words_all if w in generic_words) / max(len(words_all), 1)
    if generic_ratio > 0.6:
        generic_kw_count += 1
        if len(sample_generic) < 10:
            sample_generic.append((row['conference'], row['year'], row['title'], row['keywords']))
    else:
        specific_kw_count += 1
        if len(sample_specific) < 10:
            sample_specific.append((row['conference'], row['year'], row['title'], row['keywords']))

print(f"\nIn 500-paper sample:")
print(f"  Too generic (>60% generic words): {generic_kw_count} ({generic_kw_count/5:.1f}%)")
print(f"  Specific enough               : {specific_kw_count} ({specific_kw_count/5:.1f}%)")

print("\n--- Examples of GOOD specific keywords ---")
for conf, yr, title, kw in sample_specific[:8]:
    print(f"  [{conf} {yr}] {str(title)[:60]}")
    print(f"    {str(kw)[:120]}")

print("\n--- Examples of TOO GENERIC keywords ---")
for conf, yr, title, kw in sample_generic[:8]:
    print(f"  [{conf} {yr}] {str(title)[:60]}")
    print(f"    {str(kw)[:120]}")

# ── Conference breakdown ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("KEYWORD DIVERSITY BY CONFERENCE (avg unique words per paper)")
print(f"{'='*70}")
for conf in df_kw['conference'].dropna().unique():
    sub = df_kw[df_kw['conference'] == conf].sample(min(200, len(df_kw[df_kw['conference'] == conf])), random_state=1)
    avg_phrases = sub['keywords'].apply(lambda k: len([x for x in str(k).split(';') if x.strip()])).mean()
    avg_words   = sub['keywords'].apply(lambda k: np.mean([len(p.split()) for p in str(k).split(';') if p.strip()]) if str(k).strip() else 0).mean()
    print(f"  {conf:<12}  avg phrases: {avg_phrases:.1f}   avg words/phrase: {avg_words:.1f}")
