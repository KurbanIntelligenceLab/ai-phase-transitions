"""
Replace LDA-generated keywords with KeyBERT.
- Input:  merged_papers_lda.xlsx  (has filled abstracts; LDA rows identified by being
          originally empty/Topic: in merged_papers.xlsx)
- Output: merged_papers_final.xlsx
Rows that already had good keywords (not from LDA) are kept untouched.
"""

import logging
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_EXCEL = SCRIPT_DIR / "merged_papers_lda.xlsx"
ORIG_EXCEL  = SCRIPT_DIR / "merged_papers.xlsx"
OUTPUT_EXCEL = SCRIPT_DIR / "merged_papers_final.xlsx"

MAX_TERMS           = 12
MIN_ABSTRACT_LEN    = 40
MAX_KEYBERT_DOC_CHARS = 8000

KEYWORD_STOPLIST = {
    "language", "models", "model", "learning", "neural", "network", "networks",
    "data", "text", "image", "images", "video", "videos", "detection", "segmentation",
    "object", "objects", "based", "method", "methods", "approach", "approaches",
    "training", "deep", "representation", "feature", "features", "visual", "vision",
    "natural", "processing", "recognition", "classification", "prediction", "inference",
    "optimization", "attention", "transformer", "transformers", "embedding", "embeddings",
    "multi", "single", "large", "small", "new", "using", "used", "different",
    "multiple", "various", "effective", "efficient", "framework", "algorithm", "system",
    "task", "tasks", "semeval", "shared", "workshop", "challenge", "competition",
}

EXCLUDE_PATTERNS = [
    re.compile(r"semeval\s+\d{4}\s+task", re.I),
    re.compile(r"semeval\s+task", re.I),
    re.compile(r"sem[e\-]?eval\s+\d*\s*task", re.I),
    re.compile(r"\d{4}\s+task\s+\d+", re.I),
    re.compile(r"\d{4}\s+task\b", re.I),
]

_KEYBERT_MODEL = None

def _get_keybert_model():
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is not None:
        return _KEYBERT_MODEL
    from keybert import KeyBERT
    _KEYBERT_MODEL = KeyBERT()
    return _KEYBERT_MODEL

def _is_excluded_phrase(phrase):
    pl = phrase.lower().strip()
    return any(pat.search(pl) for pat in EXCLUDE_PATTERNS)

def _filter_phrases(phrases):
    kept = []
    seen_lower = set()
    for p in sorted(phrases, key=lambda x: (-len(x.split()), x)):
        p = p.strip()
        if not p:
            continue
        pl = p.lower()
        if _is_excluded_phrase(p):
            continue
        if len(pl.split()) == 1 and pl in KEYWORD_STOPLIST:
            continue
        if any(pl in k and pl != k for k in seen_lower):
            continue
        kept.append(p)
        seen_lower.add(pl)
    return kept

def _abstract_str(abstract):
    if abstract is None:
        return ""
    if isinstance(abstract, float) and pd.isna(abstract):
        return ""
    s = str(abstract).strip()
    return "" if not s or s.lower() == "nan" else s

def document_for_keybert(title, abstract):
    title = (title or "").strip()
    abst  = _abstract_str(abstract)
    if len(abst) >= MIN_ABSTRACT_LEN:
        body = abst[:MAX_KEYBERT_DOC_CHARS]
        return f"{title}. {body}" if title else body
    return title

def keywords_from_keybert(title, abstract=None, model=None):
    doc = document_for_keybert(title, abstract)
    if not doc or len(doc.strip()) < 5:
        return ""
    kw_model = model or _get_keybert_model()
    import inspect
    use_mmr = "use_mmr" in inspect.signature(kw_model.extract_keywords).parameters
    kwargs = {
        "keyphrase_ngram_range": (1, 3),
        "stop_words": "english",
        "top_n": MAX_TERMS * 2,
    }
    if use_mmr:
        kwargs["use_mmr"] = True
        kwargs["diversity"] = 0.7
    try:
        items = kw_model.extract_keywords(doc, **kwargs) or []
    except Exception:
        return ""
    result = []
    for item in items:
        result.append(str(item[0]).strip() if isinstance(item, (list, tuple)) else str(item).strip())
    result = _filter_phrases(result)
    if len(result) < 3:
        kwargs["keyphrase_ngram_range"] = (1, 1)
        try:
            single = kw_model.extract_keywords(doc, **kwargs) or []
        except Exception:
            single = []
        for item in single:
            w = (str(item[0]) if isinstance(item, (list, tuple)) else str(item)).strip().lower()
            if w not in KEYWORD_STOPLIST and len(w) > 2 and w not in {r.lower() for r in result}:
                result.append(str(item[0]).strip() if isinstance(item, (list, tuple)) else item)
    return "; ".join(result[:MAX_TERMS]) if result else ""

# ── Load ──────────────────────────────────────────────────────────────────────
logging.info("Loading merged_papers_lda.xlsx …")
df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")
logging.info("  shape: %s", df.shape)

logging.info("Loading original merged_papers.xlsx to identify LDA rows …")
df_orig = pd.read_excel(ORIG_EXCEL, engine="openpyxl")
was_topic = df_orig["keywords"].astype(str).str.strip().str.startswith("Topic:")
was_empty = df_orig["keywords"].isna() | (df_orig["keywords"].astype(str).str.strip() == "")
lda_mask  = (was_topic | was_empty).values

logging.info("  LDA-generated rows to replace: %d", lda_mask.sum())

# ── Run KeyBERT on LDA rows ───────────────────────────────────────────────────
logging.info("Loading KeyBERT model …")
model = _get_keybert_model()

lda_indices = df.index[lda_mask].tolist()
n = len(lda_indices)
logging.info("Processing %d rows with KeyBERT …", n)

_illegal = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]")

for i, idx in enumerate(lda_indices):
    if i % 1000 == 0:
        logging.info("  %d / %d …", i, n)
    title    = df.at[idx, "title"]
    abstract = df.at[idx, "abstract"]
    kw = keywords_from_keybert(title, abstract=abstract, model=model)
    df.at[idx, "keywords"] = kw

logging.info("KeyBERT done. Saving …")
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(lambda s: _illegal.sub(" ", s) if isinstance(s, str) else s)

df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")

# ── Summary ───────────────────────────────────────────────────────────────────
has_kw = df["keywords"].notna() & (df["keywords"].astype(str).str.strip() != "") & (df["keywords"].astype(str).str.strip() != "nan")
logging.info("\n── Summary ──")
logging.info("  Total rows          : %d", len(df))
logging.info("  Rows with keywords  : %d", has_kw.sum())
logging.info("  Rows without keywords: %d (no abstract)", (~has_kw).sum())
logging.info("Done → %s", OUTPUT_EXCEL)
