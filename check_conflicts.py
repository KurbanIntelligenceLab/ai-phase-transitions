import pandas as pd, re

df = pd.read_excel('c:/Users/User/Desktop/paper/merged_papers_keywords_keybert.xlsx')

def parse_kw(s):
    if pd.isna(s) or not str(s).strip(): return []
    return [k.strip().lower() for k in re.split(r'[;,]', str(s)) if k.strip()]

df['kw_list'] = df['keywords'].apply(parse_kw)
rows = []
for _, row in df.iterrows():
    for kw in row['kw_list']:
        rows.append({'kw': kw, 'year': int(row['year']), 'conf': row['conference']})
kw_df = pd.DataFrame(rows)

# CONFLICT (a): relative growth per venue
print("CONFLICT (a): relative growth per venue 2017->2025")
for conf in ['ACL','CVPR','ICLR','ICML','NeurIPS']:
    sub = df[df['conference']==conf].groupby('year').size()
    c17 = int(sub.get(2017, 0))
    c25 = int(sub.get(2025, 0))
    ratio = c25/c17 if c17 > 0 else 0
    print(f"  {conf}: {c17} -> {c25} = {ratio:.1f}x")

# CONFLICTS (b)(c)(d): ACL topic counts, ALL papers
print()
print("CONFLICTS (b)(c)(d): ACL topic counts - ALL papers scope")
acl_all = kw_df[kw_df['conf']=='ACL']
targets = ['large language models','neural machine translation','named entity recognition']
for kw in targets:
    total = int((acl_all['kw']==kw).sum())
    print(f"  {kw}: total={total}")

# CONFLICTS (b)(c)(d): ACL topic counts, MAIN papers only
print()
print("CONFLICTS (b)(c)(d): ACL topic counts - MAIN papers only")
main = df[df['paper_type'].astype(str).str.strip().str.lower()=='main'].copy()
main['kw_list'] = main['keywords'].apply(parse_kw)
rows_m = []
for _, row in main.iterrows():
    for kw in row['kw_list']:
        rows_m.append({'kw': kw, 'year': int(row['year']), 'conf': row['conference']})
kw_main = pd.DataFrame(rows_m)
acl_main = kw_main[kw_main['conf']=='ACL']
for kw in targets:
    total = int((acl_main['kw']==kw).sum())
    print(f"  {kw}: total={total}")

# What scope does the verified_topic_counts_by_year.csv use?
print()
print("verified_topic_counts_by_year.csv (from rebuild_figures.py, main only):")
try:
    vcsv = pd.read_csv('c:/Users/User/Desktop/paper/topic_analysis_figures_keybert_v5/verified_topic_counts_by_year.csv', index_col=0)
    for kw in ['large language models','neural machine translation','named entity recognition']:
        if kw in vcsv.index:
            total = int(vcsv.loc[kw].sum())
            print(f"  {kw}: total={total}  row={list(vcsv.loc[kw].astype(int).values)}")
except Exception as e:
    print(f"  Error: {e}")

# Derived ratios for both scopes
print()
print("Derived ratios:")
for label, acl_use in [('ALL', acl_all), ('MAIN', acl_main)]:
    llm = int((acl_use['kw']=='large language models').sum())
    nmt = int((acl_use['kw']=='neural machine translation').sum())
    ner = int((acl_use['kw']=='named entity recognition').sum())
    r_nmt = llm/nmt if nmt else 0
    r_ner = llm/ner if ner else 0
    print(f"  [{label}] LLM={llm}, NMT={nmt}, NER={ner}  =>  LLM/NMT={r_nmt:.1f}x  LLM/NER={r_ner:.1f}x")
