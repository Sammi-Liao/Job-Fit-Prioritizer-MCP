import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "jobs.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """ 
    Initialize the database tables. 

    job table:
    - status: pending, ranked, evaluated, dismissed
    """
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                company     TEXT,
                location    TEXT,
                description TEXT,
                url         TEXT,
                salary_min  REAL,
                salary_max  REAL,
                posted_at   TEXT,
                fetched_at  TEXT NOT NULL,
                embedding         TEXT,
                similarity_score  REAL,
                llm_score         INTEGER,
                llm_reasoning     TEXT,
                recommendation    TEXT,
                strengths         TEXT,
                gaps              TEXT,
                status      TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS resume (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                content     TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
        """)


def job_exists(job_id: str) -> bool:
    """ Check if a job exists in the database. """
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row is not None


def insert_job(job: dict):
    """ Insert a job into the database. """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO jobs
                (id, title, company, location, description, url,
                 salary_min, salary_max, posted_at, fetched_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                job["id"], job["title"], job["company"], job["location"],
                job["description"], job["url"],
                job.get("salary_min"), job.get("salary_max"), job.get("posted_at"),
                datetime.utcnow().isoformat(),
            ),
        )


def get_pending_jobs(limit: int = 100) -> list[dict]:
    """ Get all pending jobs from the database. """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_ranked_jobs(limit: int = 10) -> list[dict]:
    """ Get all ranked jobs from the database. """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs WHERE status = 'ranked'
            ORDER BY similarity_score DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def count_ranked_jobs() -> int:
    """ Count the number of ranked jobs in the database. """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'ranked'"
        ).fetchone()
        return row[0]


def update_job_embedding(job_id: str, embedding: list[float], similarity_score: float):
    """ Update a job's embedding and similarity score in the database. """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET embedding = ?, similarity_score = ?, status = 'ranked'
            WHERE id = ?
            """,
            (json.dumps(embedding), similarity_score, job_id),
        )


def update_job_llm(
    job_id: str,
    score: int,
    recommendation: str,
    reasoning: str,
    strengths: list[str],
    gaps: list[str],
):
    """ Update a job's LLM score, recommendation, reasoning, strengths, and gaps in the database. """
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET llm_score = ?, recommendation = ?, llm_reasoning = ?,
                strengths = ?, gaps = ?, status = 'evaluated'
            WHERE id = ?
            """,
            (
                score, recommendation, reasoning,
                json.dumps(strengths), json.dumps(gaps),
                job_id,
            ),
        )


def update_job_status(job_id: str, status: str):
    with get_connection() as conn:
        conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))


def dismiss_current_batch() -> int:
    """ Dismiss all evaluated jobs in the current batch. """
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'dismissed' WHERE status = 'evaluated'"
        )
        return cursor.rowcount


def save_resume(content: str, embedding: list[float]):
    """ Save the resume and its embedding to the database. """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO resume (id, content, embedding, updated_at)
            VALUES (1, ?, ?, ?)
            """,
            (content, json.dumps(embedding), datetime.utcnow().isoformat()),
        )


def get_resume() -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM resume WHERE id = 1").fetchone()
        if not row:
            return None
        d = dict(row)
        d["embedding"] = json.loads(d["embedding"])
        return d


def get_evaluated_jobs() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs WHERE status = 'evaluated'
            ORDER BY llm_score DESC
            """
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["strengths"] = json.loads(d["strengths"]) if d.get("strengths") else []
            d["gaps"] = json.loads(d["gaps"]) if d.get("gaps") else []
            result.append(d)
        return result


def get_stats() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["count"] for r in rows}
