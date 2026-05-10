"""
Knowledge base builder.

Reads the golden dataset and produces a cleaned, deduplicated knowledge
base with one document per row.
"""

import hashlib
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Noise patterns to strip from answer text
# ---------------------------------------------------------------------------

# Inline citation numbers  e.g. "cases.1 The"  or  "disease.2,3 Further"
_CITATION_RE = re.compile(r'(?<=\w)\.(\d[\d,]*)\s')

# Phonetic / pronunciation guides e.g. "(ath-er-o-skler-O-sis)"
_PHONETIC_RE = re.compile(r'\([a-zA-Z]+(?:[-][a-zA-Z]+){2,}\)')

# Multimedia references
_MEDIA_RE = re.compile(
    r'(Watch the (animated )?video.*?(\.|keyboard\.))'
    r'|(To enlarge the video.*?(\.|keyboard\.))'
    r'|(To reduce the video.*?(\.|keyboard\.))'
    r'|(press the Escape.*?keyboard\.)',
    re.I | re.S
)

# Boilerplate NIH / NCI phrases
_BOILERPLATE_PHRASES = [
    re.compile(p, re.I)
    for p in [
        r'This summary section describes treatments that are being studied in clinical trials\.?'
        r'\s*It may not mention every new treatment being studied\.?',
        r'Information about (ongoing )?clinical trials is available from the NCI (Web )?[Ss]ite\.?',
        r'See this graphic for a quick overview.*?\.',
        r'Read or listen to ways some patients are coping.*?\.',
        r'For more information.*?visit.*?\.',
        r'\(Watch the.*?keyboard\.\)',
    ]
]

# Trailing / leading list-item dashes that appear as sentence starters
_DASH_LIST_RE = re.compile(r'\s*-\s+')

# Multiple spaces / non-breaking spaces
_MULTI_SPACE_RE = re.compile(r'[ \t ]{2,}')

# Broken hyphenated line-breaks  e.g. "treat-\nment"
_BROKEN_HYPHEN_RE = re.compile(r'-\s*\n\s*')

# URLs (they go into source_link, not cleaned_text)
_URL_RE = re.compile(r'https?://[^\s",)\]]+')


# ---------------------------------------------------------------------------
# Text-cleaning helpers
# ---------------------------------------------------------------------------

def _remove_duplicated_sentences(text: str) -> str:
    """Remove consecutive duplicate sentences (common in MedQuAD scraped text)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen: list[str] = []
    for s in sentences:
        s = s.strip()
        if s and (not seen or s != seen[-1]):
            seen.append(s)
    return ' '.join(seen)


def _derive_title(question: str) -> str:
    """Convert a question string into a clean document title."""
    title = re.sub(r'\?+$', '', question).strip()
    # Capitalise first letter
    return title[0].upper() + title[1:] if title else title


def clean_document_text(text: str) -> str:
    """Apply all cleaning rules to a single answer text."""
    if not isinstance(text, str):
        return ''

    # 1. Remove multimedia references
    text = _MEDIA_RE.sub(' ', text)

    # 2. Remove boilerplate phrases
    for pat in _BOILERPLATE_PHRASES:
        text = pat.sub(' ', text)

    # 3. Remove inline citation numbers (e.g. "cases.1 The" → "cases. The")
    text = _CITATION_RE.sub('. ', text)

    # 4. Remove phonetic guides
    text = _PHONETIC_RE.sub('', text)

    # 5. Remove URLs
    text = _URL_RE.sub('', text)

    # 6. Fix broken hyphenated line-breaks
    text = _BROKEN_HYPHEN_RE.sub('', text)

    # 7. Replace list-item dashes with a space (preserve sentence flow)
    text = _DASH_LIST_RE.sub(' ', text)

    # 8. Collapse multiple spaces / tabs / NBSP
    text = _MULTI_SPACE_RE.sub(' ', text)

    # 9. Normalise remaining whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Remove consecutive duplicate sentences
    text = _remove_duplicated_sentences(text)

    return text


# ---------------------------------------------------------------------------
# Knowledge base builder
# ---------------------------------------------------------------------------

def build_knowledge_base(golden: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cleaned knowledge base from the golden dataset.

    Cleaning steps applied:
      - Remove empty documents (null or blank ground_truth)
      - Clean text: strip citations, phonetics, multimedia refs,
        boilerplate, URLs, broken lines, duplicate sentences
      - Remove documents that become too short after cleaning (< 50 words)
      - Remove duplicate documents (identical cleaned_text)

    Returns a DataFrame with the knowledge base schema.
    """
    df = golden.copy()

    # --- Remove empty documents ---
    df = df[df['ground_truth'].notna()]
    df = df[df['ground_truth'].str.strip().str.len() > 0]

    # --- Clean text ---
    df['cleaned_text'] = df['ground_truth'].apply(clean_document_text)

    # --- Remove documents too short after cleaning ---
    df = df[df['cleaned_text'].str.split().str.len() >= 50]

    # --- Remove duplicate documents (same cleaned_text) ---
    df['_text_hash'] = df['cleaned_text'].apply(
        lambda t: hashlib.md5(t.encode()).hexdigest()
    )
    df = df.drop_duplicates(subset=['_text_hash'], keep='first')
    df = df.drop(columns=['_text_hash'])

    # --- Derive title from question ---
    df['title'] = df['question'].apply(_derive_title)

    # --- Build final schema ---
    df = df.reset_index(drop=True)
    kb = pd.DataFrame({
        'document_id':  ['D{:03d}'.format(i + 1) for i in range(len(df))],
        'title':        df['title'].values,
        'source':       df['source'].values,
        'medical_topic':df['focus_area'].values,
        'cleaned_text': df['cleaned_text'].values,
        'source_link':  df['context_source_id'].values,
    })

    return kb
