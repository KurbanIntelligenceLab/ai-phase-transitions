"""
Pipeline:
1. Load merged_papers.xlsx; clear keywords that start with "Topic:"
2. Fill missing abstracts from merged_papers_keywords_keybert_filled.xlsx
3. For papers that still have no keywords, run LDA topic modelling
   (replicating the paper's method: abstracts → NLTK stopwords → Gensim LDA, K=40)
   and assign top topic words as keywords.
4. Save to merged_papers_lda.xlsx
"""

import re
import logging
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_EXCEL       = SCRIPT_DIR / "merged_papers.xlsx"
FILLED_EXCEL     = SCRIPT_DIR / "merged_papers_keywords_keybert_filled.xlsx"
OUTPUT_EXCEL     = SCRIPT_DIR / "merged_papers_lda.xlsx"

# LDA parameters (matching the paper)
N_TOPICS        = 40
ALPHA           = 50 / N_TOPICS   # Griffiths & Steyvers default
ETA             = 0.01
N_ITERATIONS    = 1000
RANDOM_SEED     = 42
MIN_DF          = 10              # remove words appearing in fewer than 10 docs
TOP_WORDS_PER_TOPIC = 20          # words used to label each topic
KEYWORDS_PER_PAPER  = 12         # top keyword phrases assigned back to each paper

# ── 1. Load & clean ──────────────────────────────────────────────────────────
logging.info("Loading merged_papers.xlsx …")
df = pd.read_excel(MAIN_EXCEL, engine="openpyxl")
logging.info("  shape: %s", df.shape)

# Clear keywords that start with "Topic:"
topic_mask = df["keywords"].astype(str).str.strip().str.startswith("Topic:")
n_cleared = topic_mask.sum()
df.loc[topic_mask, "keywords"] = np.nan
logging.info("  Cleared %d 'Topic:' keyword cells.", n_cleared)

# ── 2. Fill missing abstracts from filled file ───────────────────────────────
logging.info("Loading merged_papers_keywords_keybert_filled.xlsx …")
df_filled = pd.read_excel(FILLED_EXCEL, engine="openpyxl")

# Match by title (case-insensitive strip) as the safest join key
def norm_title(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

df["_title_key"]       = df["title"].apply(norm_title)
df_filled["_title_key"] = df_filled["title"].apply(norm_title)

abstract_map = (
    df_filled.dropna(subset=["abstract"])
    .drop_duplicates(subset=["_title_key"])
    .set_index("_title_key")["abstract"]
)

missing_before = df["abstract"].isna().sum()
needs_abstract = df["abstract"].isna() & df["_title_key"].isin(abstract_map.index)
df.loc[needs_abstract, "abstract"] = df.loc[needs_abstract, "_title_key"].map(abstract_map)
missing_after = df["abstract"].isna().sum()
df.drop(columns=["_title_key"], inplace=True)
logging.info("  Abstracts filled: %d → %d missing (filled %d).",
             missing_before, missing_after, missing_before - missing_after)

# ── 3. Identify papers that still need keywords ──────────────────────────────
needs_kw = df["keywords"].isna() | (df["keywords"].astype(str).str.strip() == "")
logging.info("  Papers needing keywords: %d / %d", needs_kw.sum(), len(df))

# Among those, only ones with a usable abstract can be processed by LDA
has_abstract = df["abstract"].notna() & (df["abstract"].astype(str).str.strip().str.len() > 40)
lda_mask = needs_kw & has_abstract
logging.info("  Papers for LDA (have abstract, no keywords): %d", lda_mask.sum())

if lda_mask.sum() == 0:
    logging.info("All papers have keywords. Saving …")
    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    logging.info("Done. Saved to %s", OUTPUT_EXCEL)
    raise SystemExit(0)

# ── 4. Preprocessing ─────────────────────────────────────────────────────────
logging.info("Preprocessing abstracts for LDA …")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP = set(stopwords.words("english"))

def preprocess(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP and len(t) > 2]

lda_df = df[lda_mask].copy().reset_index(drop=False)   # keep original index
lda_df["tokens"] = lda_df["abstract"].apply(preprocess)

# Build corpus – apply min_df filter
from collections import Counter
doc_freq = Counter()
for toks in lda_df["tokens"]:
    doc_freq.update(set(toks))

vocab = {w for w, cnt in doc_freq.items() if cnt >= MIN_DF}
logging.info("  Vocabulary size after min_df=%d: %d words", MIN_DF, len(vocab))

lda_df["tokens"] = lda_df["tokens"].apply(lambda toks: [t for t in toks if t in vocab])
# Drop empty docs
valid = lda_df["tokens"].apply(len) > 0
logging.info("  Docs with ≥1 token: %d / %d", valid.sum(), len(lda_df))
lda_df = lda_df[valid].copy()

# ── 5. Build Gensim dictionary & corpus ──────────────────────────────────────
logging.info("Building Gensim dictionary and BoW corpus …")
from gensim import corpora, models
from gensim.models import CoherenceModel

dictionary = corpora.Dictionary(lda_df["tokens"].tolist())
bow_corpus  = [dictionary.doc2bow(toks) for toks in lda_df["tokens"].tolist()]
logging.info("  Dictionary: %d unique tokens, %d docs", len(dictionary), len(bow_corpus))

# ── 6. Train LDA ─────────────────────────────────────────────────────────────
logging.info("Training LDA (K=%d, iter=%d) … this may take a few minutes.", N_TOPICS, N_ITERATIONS)
lda_model = models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=N_TOPICS,
    alpha=ALPHA,
    eta=ETA,
    iterations=N_ITERATIONS,
    random_state=RANDOM_SEED,
    passes=1,
)
logging.info("LDA training complete.")

# ── 7. Print topics ───────────────────────────────────────────────────────────
logging.info("\n── Top words per topic ──")
topics_words = {}
for t_id in range(N_TOPICS):
    words = [w for w, _ in lda_model.show_topic(t_id, topn=TOP_WORDS_PER_TOPIC)]
    topics_words[t_id] = words
    logging.info("  T%02d: %s", t_id + 1, ", ".join(words[:8]))

# ── 8. Assign keywords back to each paper ────────────────────────────────────
logging.info("Assigning topic keywords to papers …")

def get_keywords_for_doc(bow):
    """Get top topic words weighted by topic proportion for a document."""
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
    # Weighted sum of word probabilities across topics
    word_weight = defaultdict(float)
    for t_id, prob in topic_dist:
        for word, w_prob in lda_model.show_topic(t_id, topn=TOP_WORDS_PER_TOPIC):
            word_weight[word] += prob * w_prob
    sorted_words = sorted(word_weight.items(), key=lambda x: -x[1])
    return "; ".join(w for w, _ in sorted_words[:KEYWORDS_PER_PAPER])

keywords_assigned = []
for i, bow in enumerate(bow_corpus):
    if i % 5000 == 0:
        logging.info("  Processing doc %d / %d …", i, len(bow_corpus))
    keywords_assigned.append(get_keywords_for_doc(bow))

lda_df["_new_keywords"] = keywords_assigned

# Write keywords back into main df using original index
for _, row in lda_df.iterrows():
    orig_idx = row["index"]
    df.at[orig_idx, "keywords"] = row["_new_keywords"]

kw_filled_by_lda = lda_df.shape[0]
logging.info("  LDA keywords assigned to %d papers.", kw_filled_by_lda)

# ── 9. Save ───────────────────────────────────────────────────────────────────
logging.info("Saving to %s …", OUTPUT_EXCEL)
_illegal = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]")
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(lambda s: _illegal.sub(" ", s) if isinstance(s, str) else s)

df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
logging.info("Done. Saved %d rows to %s", len(df), OUTPUT_EXCEL)

# ── 10. Summary ───────────────────────────────────────────────────────────────
still_missing = df["keywords"].isna() | (df["keywords"].astype(str).str.strip() == "")
logging.info("\n── Summary ──")
logging.info("  Total papers         : %d", len(df))
logging.info("  'Topic:' cells cleared: %d", n_cleared)
logging.info("  Abstracts filled      : %d", missing_before - missing_after)
logging.info("  Keywords via LDA      : %d", kw_filled_by_lda)
logging.info("  Still no keywords     : %d (no abstract available)", still_missing.sum())
