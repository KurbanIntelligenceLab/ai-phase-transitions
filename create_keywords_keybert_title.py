"""
Method 2: KeyBERT keywords from abstract when available, else from title.
Reads merged_papers.xlsx; keeps existing keywords unless missing or starting with "Topic".
For those rows: if the paper has a usable abstract, runs KeyBERT on title + abstract
(title prefixes the document for context); otherwise uses the title only.
Output: merged_papers_keywords_keybert.xlsx (for comparison with other methods).
"""

import logging
import re
from pathlib import Path

import pandas as pd

# Suppress verbose transformer loading
for _name in ("transformers", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_EXCEL = SCRIPT_DIR / "merged_papers_keywords_keybert_filled.xlsx"
OUTPUT_EXCEL = SCRIPT_DIR / "merged_papers_keywords_keybert_v8.xlsx"

MAX_TERMS = 12
# Minimum abstract length (chars) to treat as present for KeyBERT (skip "nan", empty, stubs)
MIN_ABSTRACT_LEN = 40
# Cap document length for speed / memory (title is prepended before this trim)
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
    "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
}

# Exclude competition/shared-task phrases and bare year tokens (not general topic names)
EXCLUDE_PATTERNS = [
    re.compile(r"semeval\s+\d{4}\s+task", re.I),
    re.compile(r"semeval\s+task", re.I),
    re.compile(r"sem[e\-]?eval\s+\d*\s*task", re.I),
    # e.g. "2025 task 11", "2023 task 10", "2025 task", "2023 task" (ACL/SemEval task IDs)
    re.compile(r"\d{4}\s+task\s+\d+", re.I),
    re.compile(r"\d{4}\s+task\b", re.I),
    # any phrase that is just a year or starts/ends with a standalone year
    re.compile(r"^\s*(201[7-9]|202[0-5])\s*$", re.I),
    re.compile(r"\b(201[7-9]|202[0-5])\b", re.I),
]

_KEYBERT_MODEL = None


def _get_keybert_model():
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is not None:
        return _KEYBERT_MODEL
    try:
        from keybert import KeyBERT
        _KEYBERT_MODEL = KeyBERT()
        return _KEYBERT_MODEL
    except Exception as e:
        print(f"  KeyBERT not available: {e}")
        return None


def _is_excluded_phrase(phrase: str) -> bool:
    """True if phrase is a competition/task name (e.g. semeval 2025 task) not a topic name."""
    pl = phrase.lower().strip()
    return any(pat.search(pl) for pat in EXCLUDE_PATTERNS)


def _filter_phrases(phrases: list) -> list:
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


def _abstract_as_string(abstract) -> str:
    """Normalize abstract cell to plain string; empty if missing or useless."""
    if abstract is None:
        return ""
    if isinstance(abstract, float) and pd.isna(abstract):
        return ""
    s = str(abstract).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def document_for_keybert(title: str, abstract) -> str:
    """
    Text passed to KeyBERT: title + abstract when abstract is long enough,
    otherwise title only.
    """
    title = (title or "").strip()
    abst = _abstract_as_string(abstract)
    if len(abst) >= MIN_ABSTRACT_LEN:
        body = abst
        if len(body) > MAX_KEYBERT_DOC_CHARS:
            body = body[:MAX_KEYBERT_DOC_CHARS]
        if title:
            return f"{title}. {body}"
        return body
    return title


def keywords_from_document_keybert(document: str, model=None) -> str:
    """Extract keywords from arbitrary document text using KeyBERT. Returns semicolon-separated string."""
    if not document or len(str(document).strip()) < 5:
        return ""
    document = str(document).strip()
    kw_model = model or _get_keybert_model()
    if kw_model is None:
        return ""
    try:
        import inspect
        use_mmr = "use_mmr" in inspect.signature(kw_model.extract_keywords).parameters
    except Exception:
        use_mmr = False
    kwargs = {
        "keyphrase_ngram_range": (1, 3),
        "stop_words": "english",
        "top_n": MAX_TERMS * 2,
    }
    if use_mmr:
        kwargs["use_mmr"] = True
        kwargs["diversity"] = 0.7
    try:
        items = kw_model.extract_keywords(document, **kwargs) or []
    except Exception:
        return ""
    result = []
    for item in items:
        if isinstance(item, (list, tuple)) and item:
            result.append(str(item[0]).strip())
        elif isinstance(item, str):
            result.append(item.strip())
    result = _filter_phrases(result)
    if len(result) < 3:
        kwargs["keyphrase_ngram_range"] = (1, 1)
        try:
            single = kw_model.extract_keywords(document, **kwargs) or []
        except Exception:
            single = []
        for item in single:
            w = (str(item[0]) if isinstance(item, (list, tuple)) else str(item)).strip().lower()
            if w not in KEYWORD_STOPLIST and len(w) > 2 and w not in {r.lower() for r in result}:
                result.append(str(item[0]).strip() if isinstance(item, (list, tuple)) else item)
    return "; ".join(result[:MAX_TERMS]) if result else ""


def keywords_from_title_keybert(title: str, model=None, abstract=None) -> str:
    """
    Extract keywords using KeyBERT from abstract (plus title prefix) when abstract is usable,
    otherwise from title only. `abstract` optional for backward compatibility.
    """
    doc = document_for_keybert(title, abstract)
    return keywords_from_document_keybert(doc, model=model)


def should_regenerate(keyw):
    if pd.isna(keyw):
        return True
    s = str(keyw).strip()
    if not s:
        return True
    if s.lower().startswith("topic"):
        return True
    return False


def main():
    print("Method 2: KeyBERT on abstract (if present) else title")
    print("Input:", INPUT_EXCEL)
    print("Output:", OUTPUT_EXCEL)
    if not INPUT_EXCEL.exists():
        print("Input file not found.")
        return
    df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")
    if "keywords" not in df.columns:
        df["keywords"] = ""
    if "title" not in df.columns:
        print("No 'title' column.")
        return
    if "abstract" not in df.columns:
        df["abstract"] = ""
    need = df.apply(lambda r: should_regenerate(r.get("keywords")), axis=1)
    n = need.sum()
    print(f"Generating keywords for {n} rows (missing or start with 'Topic')...")
    model = _get_keybert_model()
    use_abs = 0
    use_title = 0
    for idx in df.index[need]:
        title = df.at[idx, "title"]
        abstract = df.at[idx, "abstract"]
        abst = _abstract_as_string(abstract)
        if len(abst) >= MIN_ABSTRACT_LEN:
            use_abs += 1
        else:
            use_title += 1
        df.at[idx, "keywords"] = keywords_from_title_keybert(title, model=model, abstract=abstract)
    print(f"  Source: {use_abs} rows used title+abstract, {use_title} rows title-only.")
    _illegal = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda s: _illegal.sub(" ", s) if isinstance(s, str) else s)
    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"Done. Wrote {len(df)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
