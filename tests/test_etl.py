import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medqa.transform import transform_medquad


def test_transform_removes_empty_rows() -> None:
    df = pd.DataFrame(
        [
            {"question": "What is asthma?", "answer": "A lung condition.", "source": "NIH", "focus_area": "Respiratory"},
            {"question": "", "answer": "Missing question.", "source": "NIH", "focus_area": "Respiratory"},
            {"question": "What is asthma?", "answer": "A lung condition.", "source": "NIH", "focus_area": "Respiratory"},
        ]
    )

    result = transform_medquad(df)
    assert len(result) == 1
    assert result.iloc[0]["normalized_question"] == "what is asthma"
    assert result.iloc[0]["focus_area"] == "Respiratory"


def test_transform_normalizes_source() -> None:
    df = pd.DataFrame(
        [{"question": "Why cough?", "answer": "Because of irritation.", "source": None, "focus_area": None}]
    )
    result = transform_medquad(df)
    assert result.iloc[0]["source"] == "unknown"
    assert result.iloc[0]["focus_area"] == "general"
