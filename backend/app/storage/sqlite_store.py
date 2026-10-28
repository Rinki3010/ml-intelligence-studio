import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from app.core.config import DATABASE_PATH

def connect():
    conn = sqlite3.connect(DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    with connect() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            pipeline_id TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            response_json TEXT NOT NULL
        )''')

def save_run(payload: dict[str, Any]):
    with connect() as conn:
        conn.execute('''INSERT OR REPLACE INTO pipeline_runs
            (run_id,pipeline_id,status,started_at,finished_at,duration_seconds,response_json)
            VALUES (?,?,?,?,?,?,?)''', (
            payload['run_id'], payload['pipeline_id'], payload['status'], payload['started_at'],
            payload['finished_at'], payload['duration_seconds'], json.dumps(payload)
        ))

def get_run(run_id: str):
    with connect() as conn:
        row = conn.execute('SELECT response_json FROM pipeline_runs WHERE run_id=?', (run_id,)).fetchone()
        return json.loads(row['response_json']) if row else None

def list_runs(limit: int = 20):
    with connect() as conn:
        rows = conn.execute('SELECT response_json FROM pipeline_runs ORDER BY started_at DESC LIMIT ?', (limit,)).fetchall()
        return [json.loads(row['response_json']) for row in rows]
