"""
Create/update keywords from paper titles only.
Reads merged_papers.xlsx; keeps existing keywords unless missing or starting with "Topic".
For those rows, generates topic names and methods from the title (no "Topic:" prefix).
Writes merged_papers_with_keywords.xlsx for topic prominence analysis.
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_EXCEL = SCRIPT_DIR / "merged_papers.xlsx"
OUTPUT_EXCEL = SCRIPT_DIR / "merged_papers_with_keywords_keybert.xlsx"

# Known topic/method phrases to detect in titles (case-insensitive); order longer first
KNOWN_PHRASES = [
    # Topics / tasks
    "Few-Shot Learning", "Zero-Shot Learning", "Metric Learning", "Active Learning",
    "Semantic Segmentation", "Semantic Image Segmentation", "Image Segmentation", "Instance Segmentation", "Panoptic Segmentation",
    "Action Recognition", "Action Detection", "Video Action", "Point Cloud", "Point Cloud Registration",
    "Stereo Matching", "View Synthesis", "View Extrapolation", "Shape Representation",
    "Neural Architecture Search", "Architecture Search", "Reinforcement Learning", "Deep Reinforcement Learning",
    "Domain Adaptation", "Transfer Learning", "Self-Supervised Learning", "Semi-Supervised Learning",
    "Object Detection", "3D Detection", "Pose Estimation", "Depth Estimation", "Optical Flow",
    "Image Classification", "Few-Shot", "Zero-Shot", "Multi-View", "Single View",
    "RGB-D", "SLAM", "Visual SLAM", "Bundle Adjustment", "Feature Correspondence",
    "Hair Capture", "Scene Completion", "Image Inpainting", "View Inpainting",
    "Confidence Estimation", "High-Confidence Predictions", "Uncertainty", "Robustness", "Adversarial",
    "ReLU Networks", "Graph Neural Network", "GNN", "Denoising Autoencoders", "Autoencoder",
    "Training Data", "Multiple Choice Learning", "Dynamics Generalization", "Trajectory-wise",
    "Kervolutional", "Convolutional Networks", "Deep Networks", "ReLU Networks",
    "Augmentation", "Data Augmentation", "AutoAugment", "Learning Augmentation",
    "Stereo", "Stereo Confidence", "Multiplane Images", "Signed Distance Functions",
    "Progressive Learning", "Spatio-Temporal", "Complex Action", "Video Action Detection",
    "Transformer", "Attention", "Self-Attention",
    # Method-style (often in titles)
    "Neural Network", "Neural Networks", "Deep Network", "Deep Networks",
    "Auto-DeepLab", "Guided Aggregation", "Bundle Adjusted", "Direct RGB-D",
    "Structure From Motion", "Multi-View Geometry", "Volume-Guided", "Progressive View",
    "Video Action Transformer", "Timeception", "Spatio-Temporal Progressive",
    "Locally Adaptive Fusion", "Mining Reliable Neighbors", "Coordinate-Free",
    "Semidefinite-Based", "Randomized Approach", "Carlsson-Weinshall",
    "Category Traversal", "Edge-Labeling", "Classification Weights", "GNN Denoising",
    "Hierarchical Neural Architecture Search", "Hierarchical Prediction",
    "Fourier Basis", "Computational Resource Utilization", "Hardness-Aware",
    "Learning Loss", "Uncertainty", "Learning Augmentation Strategies",
    "SDRSAC", "BAD SLAM", "GA-Net", "LAF-Net", "NM-Net", "DeepSDF",
    # Vision-language / NLP
    "Large Vision-Language Models", "LVLMs", "Vision-Language", "Vision-Language Pre-Training",
    "Language-Contrastive Decoding", "LCD", "Hallucinations", "Scene Text Detectors",
    "Tool Learning", "Large Language Models", "Large-Scale Benchmarking", "Arabic Dialects", "NLP",
    "Common Assumptions",
]

# Single words to drop when they appear alone (keep as part of phrase)
STOP_SINGLE = {
    "learning", "network", "networks", "deep", "neural", "data", "method", "approach",
    "using", "for", "with", "from", "and", "the", "via", "by", "to", "in", "on",
    "based", "new", "real", "time", "high", "low", "multi", "single", "end", "to",
}

# Normalize phrase for matching: lowercase, collapse spaces/hyphens
def _norm(s: str) -> str:
    return re.sub(r"[\s\-]+", " ", s.lower().strip())


def _split_abbreviation_in_parentheses(phrase: str) -> list:
    """If phrase ends with (ABBR) or contains (ABBR), return [phrase without parens, ABBR]; else [phrase]. So 'X (LCD)' -> ['X', 'LCD']."""
    out = []
    # Match (Abbreviation) at end: 2-15 chars, mostly letters/numbers
    m = re.search(r"\s*\(([A-Za-z0-9]{2,15})\)\s*$", phrase)
    if m:
        abbr = m.group(1).strip()
        base = phrase[: m.start()].strip()
        if base:
            out.append(base)
        if abbr:
            out.append(abbr)
        return out if out else [phrase]
    return [phrase]


# Build normalized set for lookup (original form for output)
PHRASE_NORM_TO_ORIGINAL = {}
for p in KNOWN_PHRASES:
    PHRASE_NORM_TO_ORIGINAL[_norm(p)] = p


def _extract_known_phrases(title: str) -> list:
    """Find known topic/method phrases in title (case-insensitive)."""
    if not title or len(title) < 5:
        return []
    found = []
    t_norm = _norm(title)  # same normalization as phrases (spaces, lowercase)
    for norm_p, orig in sorted(PHRASE_NORM_TO_ORIGINAL.items(), key=lambda x: -len(x[0])):
        if norm_p in t_norm and orig not in found:
            found.append(orig)
    return found


def _extract_capitalized_phrases(title: str, max_phrase_words: int = 4) -> list:
    """Extract 2-4 word sequences that look like technical terms. Tokenize by space so hyphenated compounds stay (e.g. Trajectory-wise, High-Confidence). Skip sentence fragments (Why..., How to...)."""
    if not title:
        return []
    # Tokenize by whitespace only so "Trajectory-wise", "High-Confidence" stay as one token
    tokens = title.split()
    if not tokens:
        return []
    # Sentence starters and verbs/fragment starters: don't start a phrase with these
    no_start = {"why", "how", "what", "when", "where", "and", "or", "the", "for", "with", "from", "via", "by", "to", "in", "on", "a", "an", "yield", "mitigate", "revealing", "pushing", "striking", "networks", "towards", "of", "revisiting", "about"}
    no_end = {"the", "to", "and", "from", "for", "in", "on", "a", "an", "far", "away", "about", "of", "on", "towards"}
    out = []
    i = 0
    while i < len(tokens):
        # Skip past sentence starters so we don't start a phrase with "Why" or "How to"
        if tokens[i].lower() in no_start:
            i += 1
            continue
        if i + 1 < len(tokens) and tokens[i].lower() == "how" and tokens[i + 1].lower() == "to":
            i += 2
            continue
        phrase_words = []
        j = i
        broke_on_colon = False
        while j < len(tokens) and len(phrase_words) < max_phrase_words:
            w = tokens[j]
            w_clean = w.rstrip(":,;")
            w_lower = w_clean.lower() if w_clean else ""
            if w == ":" or (w.endswith(":") and w_clean):
                if w_clean:
                    phrase_words.append(w_clean)
                if phrase_words:
                    single = " ".join(phrase_words) if len(phrase_words) > 1 else phrase_words[0]
                    if len(single) > 1 and single not in out:
                        if len(phrase_words) == 1 and (single.isupper() or "-" in single or re.match(r"^[A-Z][a-z]", single)):
                            out.append(single)
                        elif len(phrase_words) >= 2 and len(single) > 4:
                            out.append(single)
                j += 1
                broke_on_colon = True
                break
            if w_lower in no_start and phrase_words:
                break
            if w_clean:
                phrase_words.append(w_clean)
            j += 1
        if not broke_on_colon:
            if len(phrase_words) >= 2:
                if phrase_words[-1].lower() not in no_end:
                    p = " ".join(phrase_words)
                    if len(p) > 4 and p not in out:
                        out.append(p)
            elif len(phrase_words) == 1:
                w = phrase_words[0]
                if len(w) >= 2 and w not in out and (w.isupper() or re.match(r"^[A-Z][a-z]+", w) or "-" in w):
                    out.append(w)
        i = j if broke_on_colon else i + 1
    return out


def _dedupe_and_trim(phrases: list, max_terms: int = 14) -> list:
    """Deduplicate by lowercase, prefer longer; drop preposition fragments and subsumed phrases."""
    skip_start = ("for ", "by ", "with ", "from ", "and ", "to ", "in ", "on ", "via ", "the ", "why ", "how to ", "how ", "what ", "when ", "where ", "a ", "of ", "towards ", "revisiting ", "about ")
    seen_lower = set()
    result = []
    # Process shorter phrases first so we keep "ReLU Networks" and drop long fragments that contain it
    for p in sorted(phrases, key=lambda x: (len(x), x)):
        pl = p.lower()
        if pl in seen_lower:
            continue
        if len(pl.split()) == 1 and pl in STOP_SINGLE:
            continue
        if any(pl.startswith(s) for s in skip_start):
            continue
        if " for " in pl or " by " in pl or " about " in pl:
            continue
        # Skip if this phrase is a proper substring of any already kept (avoid "Neural Network" when "Graph Neural Network" is in)
        if any(pl != rl and pl in rl for rl in seen_lower):
            continue
        # If this phrase contains any already-kept as proper substring, remove the shorter and keep this (e.g. keep "Multiple Choice Learning", drop "Choice Learning")
        to_drop = [r for r in result if r.lower() != pl and r.lower() in pl]
        for r in to_drop:
            result.remove(r)
            seen_lower.discard(r.lower())
        # Skip if this phrase is a proper superset of any *remaining* kept (avoid "ReLU Networks Yield..." when "ReLU Networks" is in)
        if any(rl != pl and rl in pl for rl in seen_lower):
            continue
        seen_lower.add(pl)
        result.append(p)
        if len(result) >= max_terms:
            break
    return result


def keywords_from_title(title: str) -> str:
    """
    Generate keywords from title only: topic names and methods.
    No "Topic:" or "Methods:" prefix; semicolon-separated.
    """
    if not title or not str(title).strip():
        return ""
    title = str(title).strip()
    # 1) Known phrases
    known = _extract_known_phrases(title)
    # 2) Capitalized / technical phrases from title
    capped = _extract_capitalized_phrases(title)
    # Combine: known first, then capped that don't duplicate
    combined = list(known)
    for c in capped:
        if c not in combined and c.lower() not in {x.lower() for x in combined}:
            combined.append(c)
    # 3) Split any "X (ABBR)" into "X" and "ABBR" as separate keywords
    expanded = []
    for p in combined:
        expanded.extend(_split_abbreviation_in_parentheses(p))
    combined = _dedupe_and_trim(expanded, max_terms=14)
    return "; ".join(combined) if combined else ""


def should_regenerate_keywords(keywords_series) -> bool:
    """True if we should generate new keywords: missing or starts with 'Topic'."""
    if pd.isna(keywords_series):
        return True
    s = str(keywords_series).strip()
    if not s:
        return True
    if s.lower().startswith("topic"):
        return True
    return False


def main():
    print("Create keywords from paper titles (merged_papers.xlsx)")
    print("Keep existing keywords unless missing or starting with 'Topic'")
    print("Output:", OUTPUT_EXCEL)
    print()

    if not INPUT_EXCEL.exists():
        print(f"Input not found: {INPUT_EXCEL}")
        return

    df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")
    if "keywords" not in df.columns:
        df["keywords"] = ""
    if "title" not in df.columns:
        print("No 'title' column. Exiting.")
        return

    need = df.apply(lambda r: should_regenerate_keywords(r.get("keywords")), axis=1)
    n = need.sum()
    print(f"Regenerating keywords for {n} rows (missing or start with 'Topic')")

    for idx in df.index[need]:
        title = df.at[idx, "title"]
        kw = keywords_from_title(title)
        df.at[idx, "keywords"] = kw

    # Illegal XML chars for Excel
    _illegal = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda s: _illegal.sub(" ", s) if isinstance(s, str) else s)

    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"Done. Wrote {len(df)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
