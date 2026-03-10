import sqlite3
import hashlib
from database import get_connection, init_db

# -------------------------------------------------------------------
# Enrollment number pattern from your example:
# 2402031800008 → 24=year, 0203=college code, 18=branch+div code, 00008=student number
# We'll follow this pattern for dummy data
# -------------------------------------------------------------------

COLLEGES = ['ASOIT', 'SOCET']
BRANCHES = ['CE', 'CSE', 'IT', 'AIML', 'CYBER']
SEMESTERS = [1, 2, 3, 4, 5, 6, 7, 8]
DIVS = ['A', 'B']

# College codes for enrollment number generation
COLLEGE_CODES = {
    'ASOIT': '0203',
    'SOCET': '0204'
}

# Branch codes for enrollment number generation
BRANCH_CODES = {
    'CE':    '11',
    'CSE':   '18',
    'IT':    '21',
    'AIML':  '31',
    'CYBER': '41'
}

# Div codes for enrollment number generation
DIV_CODES = {
    'A': '0',
    'B': '5'
}

# Student first and last names for generating dummy names
FIRST_NAMES = [
    'Aarav', 'Aryan', 'Dev', 'Harsh', 'Karan',
    'Rahul', 'Rohan', 'Siddharth', 'Vivek', 'Yash',
    'Ananya', 'Diya', 'Ishita', 'Kavya', 'Neha',
    'Pooja', 'Priya', 'Riya', 'Shreya', 'Tanvi'
]

LAST_NAMES = [
    'Patel', 'Shah', 'Mehta', 'Joshi', 'Desai',
    'Sharma', 'Verma', 'Singh', 'Kumar', 'Gupta'
]

def hash_password(password):
    """Simple SHA256 hash for passwords."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_enrollment(college, branch, div, sem, student_num):
    """
    Generates a 13-digit enrollment number following the pattern:
    24 + college_code(4) + branch_code(2) + div_code(1) + sem(1) + student_num(5)
    Example: 2402031800008
    """
    college_code = COLLEGE_CODES[college]   # 4 digits
    branch_code  = BRANCH_CODES[branch]     # 2 digits
    div_code     = DIV_CODES[div]           # 1 digit
    num_str      = str(student_num).zfill(4) # 4 digits

    # 24(2) + college(4) + branch(2) + div(1) + num(4) = 13 digits total
    # sem is NOT part of enrollment — enrollment is permanent across all sems
    enrollment = f"24{college_code}{branch_code}{div_code}{num_str}"
    return enrollment


def populate_faculty():
    """Inserts dummy faculty members — one per branch per college."""
    conn = get_connection()
    cursor = conn.cursor()

    faculty_data = []
    faculty_id = 1

    for college in COLLEGES:
        for branch in BRANCHES:
            name     = f"Prof. {branch} Faculty ({college})"
            email    = f"{branch.lower()}.{college.lower()}@university.ac.in"
            password = hash_password("faculty123")  # default password for all

            faculty_data.append((name, email, password, college))

    try:
        cursor.executemany('''
            INSERT OR IGNORE INTO faculty (name, email, password, college)
            VALUES (?, ?, ?, ?)
        ''', faculty_data)
        conn.commit()
        print(f"[POPULATE] Inserted {cursor.rowcount} faculty records.")
    except Exception as e:
        print(f"[ERROR] Faculty insert failed: {e}")
    finally:
        conn.close()


def populate_students():
    """
    Inserts dummy students.
    For each college → branch → semester → div → 5 students
    Total: 2 colleges × 5 branches × 8 sems × 2 divs × 5 students = 800 students
    Kept small intentionally for easy testing.
    """
    conn = get_connection()
    cursor = conn.cursor()

    students_data = []
    name_index = 0

    for college in COLLEGES:
        for branch in BRANCHES:
            for sem in SEMESTERS:
                for div in DIVS:
                    for i in range(1, 6):  # 5 students per div per sem
                        first = FIRST_NAMES[name_index % len(FIRST_NAMES)]
                        last  = LAST_NAMES[name_index % len(LAST_NAMES)]
                        name  = f"{first} {last}"

                        roll_number       = str(i + (0 if div == 'A' else 20)).zfill(3)
                        enrollment_number = generate_enrollment(college, branch, div, sem, name_index + 1)

                        students_data.append((
                            name,
                            roll_number,
                            enrollment_number,
                            college,
                            branch,
                            sem,
                            div
                        ))

                        name_index += 1

    try:
        cursor.executemany('''
            INSERT OR IGNORE INTO students
                (name, roll_number, enrollment_number, college, branch, semester, div)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', students_data)
        conn.commit()
        print(f"[POPULATE] Inserted {cursor.rowcount} student records.")
    except Exception as e:
        print(f"[ERROR] Student insert failed: {e}")
    finally:
        conn.close()


def verify_data():
    """Quick check — prints count of records in each table."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM faculty")
    print(f"[VERIFY] Faculty count  : {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM students")
    print(f"[VERIFY] Students count : {cursor.fetchone()[0]}")

    # Show a few sample students
    print("\n[VERIFY] Sample students:")
    cursor.execute("SELECT name, roll_number, enrollment_number, college, branch, semester, div FROM students LIMIT 5")
    for row in cursor.fetchall():
        print(f"  {row['name']} | Roll: {row['roll_number']} | Enroll: {row['enrollment_number']} | {row['college']} {row['branch']} Sem{row['semester']} Div{row['div']}")

    conn.close()


if __name__ == "__main__":
    print("[POPULATE] Initializing database...")
    init_db()

    print("[POPULATE] Populating faculty...")
    populate_faculty()

    print("[POPULATE] Populating students...")
    populate_students()

    print("\n[POPULATE] Done! Verifying data...")
    verify_data()