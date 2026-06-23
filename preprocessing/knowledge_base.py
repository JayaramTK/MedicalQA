"""
Build the knowledge base from enriched data.

Cleans answer text (removes multimedia refs, boilerplate, citations, URLs, etc.),
deduplicates on MD5 hash of cleaned_text, and saves CSV + Parquet.
"""
import hashlib
import re

import pandas as pd

from .config import KNOWLEDGE_BASE_CSV, KNOWLEDGE_BASE_PARQUET

_CITATION_RE    = re.compile(r"(?<=\w)\.(\d[\d,]*)\s")
_PHONETIC_RE    = re.compile(r"\([a-zA-Z]+(?:[-][a-zA-Z]+){2,}\)")
_URL_RE         = re.compile(r"https?://[^\s\",)\]]+")
_BROKEN_HYPH_RE = re.compile(r"-\s*\n\s*")
_DASH_LIST_RE   = re.compile(r"\s*-\s+")
_MULTI_SPACE_RE = re.compile(r"[ \t\xa0]{2,}")
_MEDIA_RE = re.compile(
    r"(Watch the (animated )?video.*?(\.|keyboard\.))"
    r"|(To enlarge the video.*?(\.|keyboard\.))"
    r"|(To reduce the video.*?(\.|keyboard\.))"
    r"|(press the Escape.*?keyboard\.)",
    re.I | re.S,
)
_BOILERPLATE = [
    re.compile(p, re.I) for p in [
        r"This summary section describes treatments that are being studied.*?studied\.",
        r"Information about (ongoing )?clinical trials is available from the NCI (Web )?[Ss]ite\.?",
        r"See this graphic for a quick overview.*?\.",
        r"Read or listen to ways some patients are coping.*?\.",
        r"For more information.*?visit.*?\.",
        r"\(Watch the.*?keyboard\.\)",
    ]
]


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _MEDIA_RE.sub(" ", text)
    for pat in _BOILERPLATE:
        text = pat.sub(" ", text)
    text = _CITATION_RE.sub(". ", text)
    text = _PHONETIC_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _BROKEN_HYPH_RE.sub("", text)
    text = _DASH_LIST_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences, seen = re.split(r"(?<=[.!?])\s+", text), []
    for s in sentences:
        s = s.strip()
        if s and (not seen or s != seen[-1]):
            seen.append(s)
    return " ".join(seen)


def _derive_title(question: str) -> str:
    title = re.sub(r"\?+$", "", question).strip()
    return title[0].upper() + title[1:] if title else title


def build_knowledge_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    source_col = "ground_truth" if "ground_truth" in df.columns else "answer"
    df["cleaned_text"] = df[source_col].apply(_clean_text)
    df = df[df["cleaned_text"].str.split().str.len() >= 50]
    df["_hash"] = df["cleaned_text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.drop_duplicates(subset=["_hash"], keep="first").drop(columns=["_hash"])
    df["title"] = df["question"].apply(_derive_title)
    df = df.reset_index(drop=True)

    kb = pd.DataFrame({
        "document_id":   [f"D{i+1:03d}" for i in range(len(df))],
        "title":         df["title"].values,
        "source":        df["source"].values,
        "medical_topic": df["focus_area"].values,
        "cleaned_text":  df["cleaned_text"].values,
        "source_link":   df["context_source_id"].values,
    })
    print(f"  [KnowledgeBase] {len(kb):,} documents")
    return kb


def save(kb: pd.DataFrame) -> None:
    for path in (KNOWLEDGE_BASE_CSV, KNOWLEDGE_BASE_PARQUET):
        path.parent.mkdir(parents=True, exist_ok=True)
    kb.to_csv(KNOWLEDGE_BASE_CSV, index=False)
    kb.to_parquet(KNOWLEDGE_BASE_PARQUET, index=False)
    print(f"  [KnowledgeBase] Saved → {KNOWLEDGE_BASE_CSV}")
    print(f"  [KnowledgeBase] Saved → {KNOWLEDGE_BASE_PARQUET}")


if __name__ == "__main__":
    from .extract import extract
    from .clean import clean
    from .enrich import enrich
    save(build_knowledge_base(enrich(clean(extract()))))
