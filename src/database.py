import sqlite3

DB_NAME = "adr_predictions.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sex TEXT,
        dose INTEGER,
        polypharmacy INTEGER,
        liver_disease INTEGER,
        sert_protein TEXT,
        p_gp_activity TEXT,
        gut_microbiome TEXT,
        epigenetic_silencing REAL,
        adr_score REAL,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()
