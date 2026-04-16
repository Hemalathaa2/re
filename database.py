import sqlite3
import json

def init_db():
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        final_score REAL,
        semantic_score REAL,
        skill_score REAL,
        experience_score REAL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        result TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_result(r):
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO results (name, final_score, semantic_score, skill_score, experience_score)
    VALUES (?, ?, ?, ?, ?)
    """, (
        r["name"],
        r["final_score"],
        r["semantic_score"],
        r["skill_score"],
        r["experience_score"]
    ))

    conn.commit()
    conn.close()

def save_job(job_id, results):
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()

    c.execute("""
    INSERT OR REPLACE INTO jobs (job_id, result)
    VALUES (?, ?)
    """, (job_id, json.dumps(results)))

    conn.commit()
    conn.close()

def get_job(job_id):
    conn = sqlite3.connect("candidates.db")
    c = conn.cursor()

    c.execute("SELECT result FROM jobs WHERE job_id=?", (job_id,))
    row = c.fetchone()

    conn.close()

    return json.loads(row[0]) if row else None
