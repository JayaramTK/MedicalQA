# Architecture for Medical QA Golden Dataset

## Overview

This project separates the pipeline into three logical layers:

1. Extraction
   - Read the raw MedQuAD CSV from `data/raw/` or repository root.
2. Transformation
   - Normalize question and answer text
   - Deduplicate duplicate QA pairs
   - Populate metadata fields
3. Load
   - Persist a golden dataset artifact
   - Build a searchable knowledge database with SQLite + FTS

## Data flow

- `scripts/build_golden_dataset.py`
  - `src.medqa.extract.extract_medquad`
  - `src.medqa.transform.transform_medquad`
  - `src.medqa.load.save_golden_dataset`

- `scripts/build_knowledge_db.py`
  - Reuses the transformation pipeline
  - Builds an SQLite knowledge database in `data/knowledge/`

## Knowledge database

The knowledge base is intentionally lightweight and portable:

- `documents` table stores each QA record
- `documents_fts` virtual FTS5 table supports fast full-text retrieval
- `KnowledgeDatabase.query()` performs text matching across question, answer, source, and focus area

## Extension points

- Add vector embeddings for semantic retrieval
- Add a production search API layer
- Add automated QA prompt generation
