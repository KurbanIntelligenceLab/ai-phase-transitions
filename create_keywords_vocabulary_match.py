"""
Method 3: Controlled vocabulary + title matching.
Reads merged_papers.xlsx; keeps existing keywords unless missing or starting with "Topic".
For those rows, assigns keywords by matching the title against a fixed list of topic/method phrases.
Output: merged_papers_keywords_vocabulary.xlsx (for comparison with other methods).
"""

import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_EXCEL = SCRIPT_DIR / "merged_papers.xlsx"
OUTPUT_EXCEL = SCRIPT_DIR / "merged_papers_keywords_vocabulary.xlsx"

MAX_TERMS = 14

# Controlled vocabulary: topic and method phrases (longer first for matching).
# Normalized form (lowercase, spaces) used for matching; original form for output.
CONTROLLED_VOCABULARY = [
    "Few-Shot Learning", "Zero-Shot Learning", "Metric Learning", "Active Learning",
    "Semantic Image Segmentation", "Semantic Segmentation", "Image Segmentation", "Instance Segmentation", "Panoptic Segmentation",
    "Neural Architecture Search", "Reinforcement Learning", "Deep Reinforcement Learning",
    "Domain Adaptation", "Transfer Learning", "Self-Supervised Learning", "Semi-Supervised Learning",
    "Object Detection", "Pose Estimation", "Depth Estimation", "Optical Flow",
    "Point Cloud Registration", "Point Cloud", "Stereo Matching", "View Synthesis", "View Extrapolation",
    "RGB-D", "SLAM", "Visual SLAM", "Bundle Adjustment", "Feature Correspondence",
    "Graph Neural Network", "GNN", "Transformer", "Attention", "Self-Attention",
    "Large Vision-Language Models", "Vision-Language Pre-Training", "Vision-Language",
    "Large Language Models", "Tool Learning", "Scene Text Detectors",
    "Data Augmentation", "AutoAugment", "Neural Network", "Neural Networks",
    "Image Classification", "Action Recognition", "Action Detection", "Video Action Detection",
    "Image Inpainting", "Uncertainty", "Robustness", "Adversarial",
    "ReLU Networks", "Training Data", "Multiple Choice Learning", "Dynamics Generalization",
    "Convolutional Networks", "Deep Networks", "Signed Distance Functions",
    "Spatio-Temporal", "Complex Action", "Progressive Learning",
    "NLP", "Arabic Dialects", "Common Assumptions", "Hallucinations",
    "Language-Contrastive Decoding", "LCD", "LVLMs",
    "Large-Scale Benchmarking", "Structure From Motion", "Multi-View Geometry",
    "3D Detection", "Few-Shot", "Zero-Shot", "Multi-View",
]


def _norm(s: str) -> str:
    """Normalize for matching: lowercase, collapse spaces and hyphens."""
    return re.sub(r"[\s\-]+", " ", s.lower().strip())


def _build_vocab_lookup():
    """Return list of (normalized_phrase, original_phrase) sorted by length desc."""
    pairs = [(_norm(p), p) for p in CONTROLLED_VOCABULARY if p.strip()]
    return sorted(pairs, key=lambda x: -len(x[0]))


_VOCAB_LOOKUP = None


def _get_vocab_lookup():
    global _VOCAB_LOOKUP
    if _VOCAB_LOOKUP is None:
        _VOCAB_LOOKUP = _build_vocab_lookup()
    return _VOCAB_LOOKUP


def keywords_from_title_vocabulary(title: str) -> str:
    """Match title against controlled vocabulary. Returns semicolon-separated matches (longer first, no duplicates)."""
    if not title or len(str(title).strip()) < 3:
        return ""
    title_norm = _norm(title)
    lookup = _get_vocab_lookup()
    seen_norm = set()
    result = []
    for norm_phrase, original in lookup:
        if norm_phrase in seen_norm:
            continue
        if norm_phrase in title_norm:
            result.append(original)
            seen_norm.add(norm_phrase)
            if len(result) >= MAX_TERMS:
                break
    return "; ".join(result) if result else ""


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
    print("Method 3: Controlled vocabulary + title matching")
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
    need = df.apply(lambda r: should_regenerate(r.get("keywords")), axis=1)
    n = need.sum()
    print(f"Assigning keywords for {n} rows (missing or start with 'Topic')...")
    for idx in df.index[need]:
        df.at[idx, "keywords"] = keywords_from_title_vocabulary(df.at[idx, "title"])
    _illegal = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda s: _illegal.sub(" ", s) if isinstance(s, str) else s)
    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"Done. Wrote {len(df)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
