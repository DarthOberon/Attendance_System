import sqlite3
import hashlib
from database import get_connection, init_db

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

COLLEGES  = ['ASOIT', 'SOCET']
BRANCHES  = ['CE', 'CSE', 'IT', 'AIML', 'CYBER']
SEMESTERS = [1, 2, 3, 4, 5, 6, 7, 8]
DIVS      = ['A', 'B']

STUDENTS_PER_DIV = 20  # 20 per div = 40 per class

COLLEGE_CODES = {'ASOIT': '0203', 'SOCET': '0204'}
BRANCH_CODES  = {'CE': '11', 'CSE': '18', 'IT': '21', 'AIML': '31', 'CYBER': '41'}
DIV_CODES     = {'A': '0', 'B': '5'}

# -------------------------------------------------------------------
# REALISTIC INDIAN NAMES — equal male and female (20 each)
# -------------------------------------------------------------------

MALE_FIRST = [
    'Aarav', 'Arjun', 'Dev', 'Harsh', 'Karan',
    'Rahul', 'Rohan', 'Siddharth', 'Vivek', 'Yash',
    'Nikhil', 'Pratik', 'Raj', 'Sachin', 'Tushar',
    'Udit', 'Varun', 'Abhishek', 'Chirag', 'Dhruv'
]

FEMALE_FIRST = [
    'Ananya', 'Diya', 'Ishita', 'Kavya', 'Neha',
    'Pooja', 'Priya', 'Riya', 'Shreya', 'Tanvi',
    'Aisha', 'Bhavna', 'Chandni', 'Drashti', 'Ekta',
    'Foram', 'Gargi', 'Hetal', 'Jhanvi', 'Khushi'
]

LAST_NAMES = [
    'Patel', 'Shah', 'Mehta', 'Joshi', 'Desai',
    'Sharma', 'Verma', 'Singh', 'Kumar', 'Gupta',
    'Panchal', 'Trivedi', 'Bhatt', 'Modi', 'Chauhan',
    'Parmar', 'Solanki', 'Rana', 'Thakor', 'Makwana'
]

# -------------------------------------------------------------------
# TEAMMATES — placed in ASOIT CSE Sem 4 Div A
# Roll numbers 001-005 to put them at the top
# -------------------------------------------------------------------
TEAMMATES = [
    'Aryan Sharma',
    'Kalpit Panchal',
    'Shailesh Prajapati',
    'Nihal Bhavsar',
    'Jaimin Panchal'
]


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def generate_enrollment(college, branch, div, student_num):
    """13 digit: 24 + college(4) + branch(2) + div(1) + num(4)"""
    return f"24{COLLEGE_CODES[college]}{BRANCH_CODES[branch]}{DIV_CODES[div]}{str(student_num).zfill(4)}"


def get_name(index, div):
    """
    Returns a realistic name alternating male/female.
    First 10 slots = male, next 10 = female for each div.
    """
    if div == 'A':
        # Roll 001-010 male, 011-020 female
        if index <= 10:
            first = MALE_FIRST[(index - 1) % len(MALE_FIRST)]
        else:
            first = FEMALE_FIRST[(index - 11) % len(FEMALE_FIRST)]
    else:
        # Div B — offset to get different names
        if index <= 10:
            first = MALE_FIRST[(index + 9) % len(MALE_FIRST)]
        else:
            first = FEMALE_FIRST[(index + 9 - 10) % len(FEMALE_FIRST)]

    last = LAST_NAMES[(index - 1) % len(LAST_NAMES)]
    return f"{first} {last}"


def populate_faculty():
    conn   = get_connection()
    cursor = conn.cursor()

    faculty_data = []
    for college in COLLEGES:
        for branch in BRANCHES:
            name     = f"Prof. {branch} Faculty ({college})"
            email    = f"{branch.lower()}.{college.lower()}@university.ac.in"
            password = hash_password("faculty123")
            faculty_data.append((name, email, password, college))

    cursor.executemany('''
        INSERT OR IGNORE INTO faculty (name, email, password, college)
        VALUES (?, ?, ?, ?)
    ''', faculty_data)
    conn.commit()
    conn.close()
    print(f"[POPULATE] Faculty inserted.")


def populate_students():
    conn   = get_connection()
    cursor = conn.cursor()

    students_data = []
    global_num = 1  # for unique enrollment numbers

    for college in COLLEGES:
        for branch in BRANCHES:
            for sem in SEMESTERS:
                for div in DIVS:

                    # --- Special case: ASOIT CSE Sem 4 Div A ---
                    # Insert teammates as first 5 students
                    is_teammate_class = (
                        college == 'ASOIT' and
                        branch  == 'CSE'   and
                        sem     == 4       and
                        div     == 'A'
                    )

                    if is_teammate_class:
                        # Add 5 teammates first (roll 001-005)
                        for i, teammate_name in enumerate(TEAMMATES, start=1):
                            roll         = str(i).zfill(3)
                            enrollment   = generate_enrollment(college, branch, div, global_num)
                            students_data.append((
                                teammate_name, roll, enrollment,
                                college, branch, sem, div
                            ))
                            global_num += 1

                        # Fill remaining 15 slots with regular students (roll 006-020)
                        for i in range(6, STUDENTS_PER_DIV + 1):
                            name       = get_name(i, div)
                            roll       = str(i).zfill(3)
                            enrollment = generate_enrollment(college, branch, div, global_num)
                            students_data.append((
                                name, roll, enrollment,
                                college, branch, sem, div
                            ))
                            global_num += 1

                    else:
                        # Regular class — 20 students
                        for i in range(1, STUDENTS_PER_DIV + 1):
                            name       = get_name(i, div)
                            roll       = str(i).zfill(3)
                            enrollment = generate_enrollment(college, branch, div, global_num)
                            students_data.append((
                                name, roll, enrollment,
                                college, branch, sem, div
                            ))
                            global_num += 1

    cursor.executemany('''
        INSERT OR IGNORE INTO students
            (name, roll_number, enrollment_number, college, branch, semester, div)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', students_data)
    conn.commit()
    conn.close()
    print(f"[POPULATE] Inserted {len(students_data)} students.")


def verify_data():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM faculty")
    print(f"\n[VERIFY] Faculty  : {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM students")
    print(f"[VERIFY] Students : {cursor.fetchone()[0]}")

    # Show teammates
    print("\n[VERIFY] Teammates in ASOIT CSE Sem4 DivA:")
    cursor.execute('''
        SELECT name, roll_number, enrollment_number
        FROM students
        WHERE college='ASOIT' AND branch='CSE' AND semester=4 AND div='A'
        ORDER BY CAST(roll_number AS INTEGER)
        LIMIT 10
    ''')
    for row in cursor.fetchall():
        print(f"  Roll {row['roll_number']} | {row['name']} | {row['enrollment_number']}")

    # Show sample from another class
    print("\n[VERIFY] Sample from SOCET IT Sem2 DivB:")
    cursor.execute('''
        SELECT name, roll_number, enrollment_number
        FROM students
        WHERE college='SOCET' AND branch='IT' AND semester=2 AND div='B'
        ORDER BY CAST(roll_number AS INTEGER)
        LIMIT 5
    ''')
    for row in cursor.fetchall():
        print(f"  Roll {row['roll_number']} | {row['name']} | {row['enrollment_number']}")

    conn.close()


if __name__ == "__main__":
    print("[POPULATE] Full reset — dropping and recreating database...")

    # Full reset — delete the db file and recreate
    import os
    from database import DB_PATH
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[POPULATE] Deleted old database: {DB_PATH}")

    init_db()
    print("[POPULATE] Fresh database created.")

    print("\n[POPULATE] Inserting faculty...")
    populate_faculty()

    print("[POPULATE] Inserting students...")
    populate_students()

    verify_data()

    print("\n[POPULATE] ✅ Done! Run app.py and login with:")
    print("  Email   : cse.asoit@university.ac.in")
    print("  Password: faculty123")