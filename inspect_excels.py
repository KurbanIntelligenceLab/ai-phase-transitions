import pandas as pd

df1 = pd.read_excel('merged_papers.xlsx', engine='openpyxl')
print('=== merged_papers.xlsx ===')
print('shape:', df1.shape)
print('columns:', list(df1.columns))
print('\nkeywords sample (first 10):')
print(df1['keywords'].head(10).to_string())
print('\nkeywords starting with Topic:')
mask = df1['keywords'].astype(str).str.strip().str.startswith('Topic:')
print('count:', mask.sum())
print(df1.loc[mask, 'keywords'].head(5).to_string())
print('\nabstract null count:', df1['abstract'].isna().sum() if 'abstract' in df1.columns else 'no abstract col')

df2 = pd.read_excel('merged_papers_keywords_keybert_filled.xlsx', engine='openpyxl')
print('\n=== merged_papers_keywords_keybert_filled.xlsx ===')
print('shape:', df2.shape)
print('columns:', list(df2.columns))
print('\nabstract null count:', df2['abstract'].isna().sum() if 'abstract' in df2.columns else 'no abstract col')
print('\nfirst 3 rows (key cols):')
cols = [c for c in ['title','year','conference','keywords','abstract'] if c in df2.columns]
print(df2[cols].head(3).to_string())
