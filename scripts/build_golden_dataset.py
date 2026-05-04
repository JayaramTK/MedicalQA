import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medqa.config import DEFAULT_CONFIG
from medqa.extract import extract_medquad
from medqa.load import save_golden_dataset
from medqa.transform import sample_golden_dataset, transform_medquad


def main() -> None:
    df = extract_medquad(DEFAULT_CONFIG["raw_paths"])
    processed = transform_medquad(df)
    golden = sample_golden_dataset(processed)

    print(golden.columns)

    output_path = save_golden_dataset(golden)
    print(f"Golden dataset saved to: {output_path}")
    print(f"Total records: {len(golden)}")
    print(f"Focus areas: {golden['focus_area'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
