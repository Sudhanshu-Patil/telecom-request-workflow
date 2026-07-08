"""SQLite audit trail: every case and every remediation action is a row.

The Case Log and Dashboard tabs are plain reads over these two tables -- no
state lives anywhere else. Connections are opened per call: cheap for
SQLite, and safe under Streamlit's threading model.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    Branch,
    CaseRecord,
    CaseStatus,
    ClassificationResult,
    RemediationAction,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "cases.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id        TEXT PRIMARY KEY,
    raw_text       TEXT NOT NULL,
    classification TEXT NOT NULL,  -- ClassificationResult as JSON
    request_type   TEXT NOT NULL,  -- denormalised from classification for charts
    urgency        TEXT NOT NULL,  -- denormalised from classification for charts
    branch         TEXT NOT NULL,
    status         TEXT NOT NULL,
    sla_deadline   TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id   TEXT NOT NULL REFERENCES cases(case_id),
    step_name TEXT NOT NULL,
    detail    TEXT NOT NULL,
    artifact  TEXT,               -- JSON payload: draft text, routing notice, ...
    simulated INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_actions_case ON actions(case_id);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables if absent; safe to call on every app start."""
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(_SCHEMA)


def insert_case(case: CaseRecord, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Persist a case; request_type/urgency are denormalised for the dashboard."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO cases (case_id, raw_text, classification, request_type,
                                  urgency, branch, status, sla_deadline, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.case_id,
                case.raw_text,
                case.classification.model_dump_json(),
                case.classification.request_type.value,
                case.classification.urgency.value,
                case.branch.value,
                case.status.value,
                case.sla_deadline.isoformat() if case.sla_deadline else None,
                case.created_at.isoformat(),
            ),
        )


def update_case_status(
    case_id: str, status: CaseStatus, db_path: str | Path = DEFAULT_DB_PATH
) -> None:
    """Move a case through its lifecycle (OPEN -> RESOLVED / HUMAN_REVIEW)."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("UPDATE cases SET status = ? WHERE case_id = ?", (status.value, case_id))


def insert_action(action: RemediationAction, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Append one remediation step to the audit trail."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO actions (case_id, step_name, detail, artifact, simulated, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                action.case_id,
                action.step_name,
                action.detail,
                json.dumps(action.artifact, ensure_ascii=False, default=str)
                if action.artifact is not None
                else None,
                int(action.simulated),
                action.timestamp.isoformat(),
            ),
        )


def _row_to_case(row: sqlite3.Row) -> CaseRecord:
    return CaseRecord(
        case_id=row["case_id"],
        raw_text=row["raw_text"],
        classification=ClassificationResult.model_validate_json(row["classification"]),
        branch=Branch(row["branch"]),
        status=CaseStatus(row["status"]),
        sla_deadline=datetime.fromisoformat(row["sla_deadline"]) if row["sla_deadline"] else None,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def get_case(case_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> CaseRecord | None:
    """Load one case back into its typed form, or None if unknown."""
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return _row_to_case(row) if row else None


def list_cases(
    status: CaseStatus | str | None = None,
    branch: Branch | str | None = None,
    request_type: str | None = None,
    limit: int = 500,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Case rows as plain dicts (newest first) for the Case Log tab."""
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("status", status), ("branch", branch), ("request_type", request_type)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value.value if isinstance(value, (CaseStatus, Branch)) else value)
    sql = "SELECT * FROM cases"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, case_id DESC LIMIT ?"
    with closing(_connect(db_path)) as conn:
        return [dict(row) for row in conn.execute(sql, (*params, limit)).fetchall()]


def list_actions(
    case_id: str | None = None, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict[str, Any]]:
    """Action rows: chronological for one case, newest first across cases."""
    with closing(_connect(db_path)) as conn:
        if case_id is not None:
            cursor = conn.execute(
                "SELECT * FROM actions WHERE case_id = ? ORDER BY id ASC", (case_id,)
            )
        else:
            cursor = conn.execute("SELECT * FROM actions ORDER BY id DESC LIMIT 1000")
        return [dict(row) for row in cursor.fetchall()]


def dashboard_stats(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, dict[str, int]]:
    """Counts by request type, status and urgency for the Dashboard charts."""
    stats: dict[str, dict[str, int]] = {}
    with closing(_connect(db_path)) as conn:
        for key, column in (
            ("by_type", "request_type"),
            ("by_status", "status"),
            ("by_urgency", "urgency"),
        ):
            rows = conn.execute(
                f"SELECT {column} AS k, COUNT(*) AS n FROM cases GROUP BY {column}"
            ).fetchall()
            stats[key] = {row["k"]: row["n"] for row in rows}
    return stats
