# Golden Dataset Processing Pipeline

## Overview

The pipeline reads raw MedQuAD data and produces a curated golden dataset of 100 unique medical QA pairs, saved as both CSV and Parquet under `data/processed/`.

```
data/raw/medquad.csv
        │
        ▼
   Step 1: Basic Cleaning          →  data/processed/medquad_cleaned.csv
        │
        ▼
   Step 2: URL Filter              →  data/processed/medquad_url_filtered.csv
        │
        ▼
   Step 3: Enrichment (on cleaned)
        │
        ▼
   Step 4: Golden Sampling         →  data/processed/golden_medquad.csv
                                       data/processed/golden_medquad.parquet
```

---

## Step 1 — Basic Cleaning (`clean_medquad`)

Applied to the full raw dataset (~16 k rows).

| Rule | Detail |
|---|---|
| Drop missing question | Rows where `question` is empty or null → removed |
| Drop missing answer | Rows where `answer` is empty or null → removed |
| Drop missing focus area | Rows where `focus_area` is empty or null → removed |
| Drop short answers | Answers with **< 100 words** → removed |
| Drop noisy text | Answers where **< 70 % of characters are alphabetic** → removed |
| Deduplicate | Per `(focus_area, normalized_question)` pair; the longest answer is kept, shorter duplicates removed |
| Reset index | Index reset after all drops |

**Output:** ~9,600 clean, deduplicated rows.

---

## Step 2 — URL Filter (`filter_url_rows`) — *Intermediate artifact*

| Rule | Detail |
|---|---|
| URL presence check | Row kept only if its `answer` contains a valid `http://` or `https://` URL |
| URL extraction | First matching URL stored as `context_source_id` |

**Output:** ~45 rows saved to `medquad_url_filtered.csv` for reference.  
This file is **not** used as the sampling pool for the golden dataset — it documents which source answers contain explicit web links.

---

## Step 3 — Enrichment (`enrich_medquad`)

Applied to the full cleaned dataset from Step 1.

### question_type

Assigned by keyword match on the question text. First match wins.

| Type | Keywords matched |
|---|---|
| `symptom` | symptom, symptoms, sign, signs |
| `treatment` | treatment, treat, therapy, medication, drug, cure, manage |
| `cause` | cause, causes, caused, why, etiology |
| `Diagnosis` | diagnos*, detect, test, screening, examination |
| `Prevention` | prevent, prevention, avoid, reducing risk |
| `general` | anything else (e.g. "What is …", "Who is at risk …") |

### difficulty_level

Based on word counts of question and answer.

| Level | Condition |
|---|---|
| `easy` | question ≤ 10 words **AND** answer ≤ 200 words |
| `hard` | question > 20 words **OR** answer > 400 words |
| `medium` | all other cases |

---

## Step 4 — Golden Sampling (`sample_golden_dataset`)

Collects exactly **100 unique-question rows** from the top focus areas (ranked by available unique-question row count).

### Algorithm

1. **Rank** focus areas by number of unique questions available (descending).
2. **For each area** (in rank order, until 100 rows are collected):
   - Determine `take = min(unique_rows_in_area, 10, rows_still_needed)`.
   - **Diversity pass** — pick one row per `question_type`, starting from the rarest type first.
   - **Fill pass** — add remaining rows from the area pool without replacement until `take` is reached.
3. **Stop** once the running total hits 100.
4. After sampling, **extract `context_source_id`** from each row's answer (URL if present, else empty).
5. Assign sequential `golden_id` values (`GOLD0001` … `GOLD0100`).
6. Copy `answer` into both `ground_truth` and `expected_context`.

### Why more than 10 focus areas?

MedQuAD uses approximately 10 fixed question templates per disease (symptoms, causes, treatment, diagnosis, prevention, etc.). After deduplication, most focus areas yield 8–10 unique questions. To collect 100 rows with **no duplicate questions and no replacement**, the algorithm draws from ~12 top focus areas rather than strictly 10.

---

## Output Schema

| Column | Description |
|---|---|
| `golden_id` | Unique row identifier — `GOLD0001` to `GOLD0100` |
| `question` | Medical question from MedQuAD |
| `ground_truth` | Original MedQuAD answer |
| `source` | Source institution (e.g. NCI, NIDDK) |
| `focus_area` | Medical topic / disease name |
| `expected_context` | Evidence text — same as `ground_truth` for now |
| `context_source_id` | Website URL extracted from answer, if present |
| `question_type` | One of: symptom, treatment, cause, Diagnosis, Prevention, general |
| `difficulty_level` | One of: easy, medium, hard |
