import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows accessing columns by name like dict
    return conn


def init_db():
    """Creates all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # --- FACULTY TABLE ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faculty (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            college  TEXT    NOT NULL CHECK(college IN ('ASOIT', 'SOCET'))
        )
    ''')

    # --- STUDENTS TABLE ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT    NOT NULL,
            roll_number       TEXT    NOT NULL,
            enrollment_number TEXT    NOT NULL UNIQUE,
            college           TEXT    NOT NULL CHECK(college IN ('ASOIT', 'SOCET')),
            branch            TEXT    NOT NULL CHECK(branch IN ('CE', 'CSE', 'IT', 'AIML', 'CYBER')),
            semester          INTEGER NOT NULL CHECK(semester BETWEEN 1 AND 8),
            div               TEXT    NOT NULL
        )
    ''')

    # --- LECTURES TABLE ---
    # One record per attendance session uploaded by faculty
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lectures (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id INTEGER NOT NULL,
            college    TEXT    NOT NULL,
            branch     TEXT    NOT NULL,
            semester   INTEGER NOT NULL,
            div        TEXT    NOT NULL,
            subject    TEXT    NOT NULL,
            date       TEXT    NOT NULL,   -- stored as YYYY-MM-DD
            FOREIGN KEY (faculty_id) REFERENCES faculty(id)
        )
    ''')

    # --- ATTENDANCE TABLE ---
    # One record per student per lecture — always stores both present and absent
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            lecture_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status     TEXT    NOT NULL CHECK(status IN ('PRESENT', 'ABSENT')),
            FOREIGN KEY (lecture_id) REFERENCES lectures(id),
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized successfully at: {DB_PATH}")


if __name__ == "__main__":
    init_db()