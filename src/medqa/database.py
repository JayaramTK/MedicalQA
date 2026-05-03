import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict

from .text import normalize_question, normalize_text


@dataclass
class KnowledgeDatabase:
    path: Path
    connection: sqlite3.Connection = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT NOT NULL,
                focus_area TEXT NOT NULL,
                metadata TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                question,
                answer,
                source,
                focus_area,
                content='documents',
                content_rowid='id'
            );
            """
        )
        self.connection.commit()

    def insert_records(self, records: Iterable[Dict[str, str]]) -> None:
        insert_sql = (
            "INSERT INTO documents(question, normalized_question, answer, source, focus_area, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?)"
        )
        for record in records:
            question = normalize_text(record.get("question", ""))
            answer = normalize_text(record.get("answer", ""))
            source = normalize_text(record.get("source", "unknown"))
            focus_area = normalize_text(record.get("focus_area", "general"))
            normalized_question = normalize_question(question)
            metadata = json.dumps({"original_source": record.get("source"), "topic": focus_area})
            self.connection.execute(
                insert_sql,
                (question, normalized_question, answer, source, focus_area, metadata),
            )
        self.connection.commit()
        self._sync_fts()

    def _sync_fts(self) -> None:
        self.connection.execute(
            "INSERT INTO documents_fts(documents_fts, rowid, question, answer, source, focus_area)"
            " VALUES('delete', 0, '', '', '', '')"
        )
        self.connection.execute(
            "INSERT INTO documents_fts(rowid, question, answer, source, focus_area)"
            " SELECT id, question, answer, source, focus_area FROM documents"
        )
        self.connection.commit()

    def query(self, text: str, limit: int = 20) -> list[Dict[str, str]]:
        query_text = normalize_question(text)
        cursor = self.connection.execute(
            "SELECT d.id, d.question, d.answer, d.source, d.focus_area, d.metadata"
            " FROM documents_fts f"
            " JOIN documents d ON d.id = f.rowid"
            " WHERE documents_fts MATCH ?"
            " LIMIT ?",
            (query_text, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        if self.connection:
            self.connection.close()
