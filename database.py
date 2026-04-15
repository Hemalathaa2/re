import sqlite3

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
