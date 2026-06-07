import pandas as pd

df = pd.read_excel('merged_papers_lda.xlsx', engine='openpyxl')
print('Shape:', df.shape)

kw = df['keywords']
has_kw = kw.notna() & (kw.astype(str).str.strip() != '') & (kw.astype(str).str.strip() != 'nan')
print(f'Papers with keywords:    {has_kw.sum():>6} / {len(df)}')
print(f'Papers without keywords: {(~has_kw).sum():>6} / {len(df)}  (no abstract available)')

# Show sample LDA keywords
lda_sample = df[has_kw & df['keywords'].astype(str).str.contains(';')].head(5)
for _, row in lda_sample.iterrows():
    print(f'\n  [{row["conference"]} {row["year"]}] {str(row["title"])[:60]}')
    print(f'  keywords: {str(row["keywords"])[:120]}')

print('\nTop topic words sample (first 3 LDA keyword rows):')
mask = has_kw
print(df.loc[mask, 'keywords'].head(3).to_string())
