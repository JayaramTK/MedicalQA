# Medical Question and Answer Chatbot

This repository sets up a golden dataset and knowledge database for a medical QA chatbot based on the MedQuAD dataset.

## Goals

- Extract MedQuAD question-answer pairs
- Normalize and deduplicate the dataset to create a golden dataset
- Build a searchable medical knowledge database
- Provide clear, maintainable ETL and data architecture

## Structure

- `data/raw/` - raw imported datasets
- `data/processed/` - normalized, validated golden dataset files
- `data/knowledge/` - production knowledge database artifacts
- `src/medqa/` - ETL and knowledge database implementation
- `scripts/` - pipeline entrypoints
- `tests/` - unit tests for data processing and database construction
- `docs/` - architectural design and schema documentation
- `notebooks/` - exploration and analysis notebooks

## Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place `medquad.csv` in `data/raw/` or keep it at repository root.

3. Build the golden dataset:

```bash
python scripts/build_golden_dataset.py
```

4. Build the knowledge database:

```bash
python scripts/build_knowledge_db.py
```

## Notes

The pipeline uses a flexible file path strategy so the dataset can be loaded from the root or from `data/raw/`.
