# Data Schema

## Golden dataset schema

- `question`: original question string from MedQuAD
- `answer`: source answer text
- `source`: origin or dataset source identifier
- `focus_area`: medical topic or classification label
- `normalized_question`: lower-case normalized question used for deduplication and search

## Knowledge database schema

- `documents`
  - `id`: integer primary key
  - `question`: original question text
  - `normalized_question`: normalized text for search consistency
  - `answer`: answer text
  - `source`: dataset source metadata
  - `focus_area`: medical topic label
  - `metadata`: JSON-encoded metadata object

- `documents_fts`
  - FTS5 virtual table over `question`, `answer`, `source`, and `focus_area`
