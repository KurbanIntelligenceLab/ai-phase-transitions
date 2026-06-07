import pandas as pd
import numpy as np

df = pd.read_excel('merged_papers_lda.xlsx', engine='openpyxl')

# Identify LDA-generated keywords: rows that originally had no keywords
# LDA keywords tend to be generic single words like "learning; model; data; ..."
# KeyBERT keywords are multi-word phrases; original keywords may be author-provided phrases

# Load original to find which rows were LDA-assigned
df_orig = pd.read_excel('merged_papers.xlsx', engine='openpyxl')
df_orig['_had_topic'] = df_orig['keywords'].astype(str).str.strip().str.startswith('Topic:')
df_orig['_was_empty'] = df_orig['keywords'].isna() | (df_orig['keywords'].astype(str).str.strip() == '')
df_orig['_lda_candidate'] = df_orig['_had_topic'] | df_orig['_was_empty']

df['_lda_generated'] = df_orig['_lda_candidate']

lda_rows = df[df['_lda_generated'] == True].copy()
print(f"Total LDA-generated keyword rows: {len(lda_rows)}")

sample = lda_rows.sample(200, random_state=42)

print("\n=== 200 RANDOM LDA-KEYWORD SAMPLES ===\n")
for i, (_, row) in enumerate(sample.iterrows(), 1):
    print(f"[{i:3d}] [{row['conference']} {row['year']}] {str(row['title'])[:70]}")
    print(f"       KW: {str(row['keywords'])[:120]}")
    print()

# Also show the pattern – are they all generic?
print("\n=== KEYWORD DIVERSITY CHECK ===")
kws = sample['keywords'].dropna().tolist()
all_words = []
for k in kws:
    all_words.extend([w.strip() for w in str(k).split(';')])

from collections import Counter
counts = Counter(all_words)
print("Top 30 most common keyword tokens across 200 samples:")
for word, cnt in counts.most_common(30):
    print(f"  {word:<25} {cnt:>4}x  ({cnt/200*100:.0f}% of docs)")
