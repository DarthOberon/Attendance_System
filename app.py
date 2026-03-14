from flask import Flask, request, session, redirect, url_for, jsonify, render_template
import os
import hashlib
from datetime import date
from database import get_connection, init_db

# --- Import OCR modules ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ocr_screenshot import process_digital_screenshot
from ocr_diary_easyocr import extract_numbers
from model_grid_detector import detect_grid

app = Flask(__name__)
app.secret_key = "idt_attendac_secret_key"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'faculty_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# -------------------------------------------------------------------
# AUTH ROUTES
# -------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    if 'faculty_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    email    = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()

    if not email or not password:
        return render_template('login.html', error="Please enter email and password.")

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faculty WHERE email = ?", (email,))
    faculty = cursor.fetchone()
    conn.close()

    if not faculty or faculty['password'] != hash_password(password):
        return render_template('login.html', error="Invalid email or password.")

    session['faculty_id']   = faculty['id']
    session['faculty_name'] = faculty['name']
    session['college']      = faculty['college']

    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# -------------------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------------------

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    conn   = get_connection()
    cursor = conn.cursor()

    # Total students in this faculty's college
    cursor.execute('''
        SELECT COUNT(*) FROM students WHERE college = ?
    ''', (session['college'],))
    total_students = cursor.fetchone()[0]

    # Today's present count for this faculty
    today = str(date.today())
    cursor.execute('''
        SELECT COUNT(*) FROM attendance a
        JOIN lectures l ON a.lecture_id = l.id
        WHERE l.faculty_id = ? AND l.date = ? AND a.status = 'PRESENT'
    ''', (session['faculty_id'], today))
    today_present = cursor.fetchone()[0]

    # Attendance percentage
    today_percentage = round((today_present / total_students * 100), 1) if total_students > 0 else 0

    # Recent lectures with present count
    cursor.execute('''
        SELECT l.date, l.subject, l.branch, l.semester, l.div,
               COUNT(CASE WHEN a.status = 'PRESENT' THEN 1 END) as present_count
        FROM lectures l
        JOIN attendance a ON a.lecture_id = l.id
        WHERE l.faculty_id = ?
        GROUP BY l.id
        ORDER BY l.date DESC
        LIMIT 10
    ''', (session['faculty_id'],))
    recent_lectures = cursor.fetchall()
    conn.close()

    return render_template('dashboard.html',
        faculty_name     = session['faculty_name'],
        total_students   = total_students,
        today_present    = today_present,
        today_percentage = today_percentage,
        recent_lectures  = recent_lectures
    )


# -------------------------------------------------------------------
# UPLOAD PAGE (GET = show form, POST = run OCR only, no DB save yet)
# -------------------------------------------------------------------

@app.route('/upload', methods=['GET'])
@login_required
def upload():
    return render_template('upload.html',
        faculty_name = session['faculty_name'],
        college      = session['college'],
        branches     = ['CE', 'CSE', 'IT', 'AIML', 'CYBER'],
        semesters    = list(range(1, 9)),
        divs         = ['A', 'B']
    )


@app.route('/ocr_preview', methods=['POST'])
@login_required
def ocr_preview():
    # --- 1. GET FORM DATA ---
    college  = request.form.get('college')
    branch   = request.form.get('branch')
    semester = request.form.get('semester')
    div      = request.form.get('div')
    subject  = request.form.get('subject', '').strip()
    mode     = request.form.get('mode')
    id_type  = request.form.get('id_type')
    file     = request.files.get('image')

    # --- 2. VALIDATE ---
    if not all([college, branch, semester, div, subject, mode, id_type, file]):
        return render_template('upload.html',
            error        = "All fields are required.",
            faculty_name = session['faculty_name'],
            college      = session['college'],
            branches     = ['CE', 'CSE', 'IT', 'AIML', 'CYBER'],
            semesters    = list(range(1, 9)),
            divs         = ['A', 'B']
        )

    if not allowed_file(file.filename):
        return render_template('upload.html',
            error        = "Only PNG, JPG, JPEG files are allowed.",
            faculty_name = session['faculty_name'],
            college      = session['college'],
            branches     = ['CE', 'CSE', 'IT', 'AIML', 'CYBER'],
            semesters    = list(range(1, 9)),
            divs         = ['A', 'B']
        )

    # --- 3. SAVE IMAGE ---
    filename = f"{session['faculty_id']}_{mode}_{date.today()}.{file.filename.rsplit('.', 1)[1].lower()}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # --- 4. RUN OCR ONLY (no DB save yet) ---
    try:
        if mode == 'screenshot':
            ocr_numbers = process_digital_screenshot(filepath)
        elif mode == 'diary':
            ocr_numbers = extract_numbers(filepath)
        elif mode == 'sheet':
            ocr_numbers = detect_grid(filepath)
        else:
            ocr_numbers = []
    except Exception as e:
        return render_template('upload.html',
            error        = f"OCR failed: {str(e)}",
            faculty_name = session['faculty_name'],
            college      = session['college'],
            branches     = ['CE', 'CSE', 'IT', 'AIML', 'CYBER'],
            semesters    = list(range(1, 9)),
            divs         = ['A', 'B']
        )

    # --- 5. MATCH OCR NUMBERS TO STUDENTS ---
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, roll_number, enrollment_number
        FROM students
        WHERE college = ? AND branch = ? AND semester = ? AND div = ?
    ''', (college, branch, int(semester), div))
    all_students = cursor.fetchall()
    conn.close()

    roll_map       = {s['roll_number']: s for s in all_students}
    enrollment_map = {s['enrollment_number']: s for s in all_students}

    matched   = []
    unmatched = []

    for num in ocr_numbers:
        num = str(num).strip()
        if id_type == 'roll':
            normalized = num.zfill(3)
            if normalized in roll_map:
                matched.append(normalized)
            else:
                unmatched.append(num)
        elif id_type == 'enrollment':
            if num in enrollment_map:
                matched.append(num)
            else:
                unmatched.append(num)

    # --- 6. STORE FORM DATA IN SESSION for confirm step ---
    session['pending'] = {
        'college' : college,
        'branch'  : branch,
        'semester': semester,
        'div'     : div,
        'subject' : subject,
        'mode'    : mode,
        'id_type' : id_type,
        'matched' : matched,
        'filename': filename
    }

    return render_template('ocr_preview.html',
        faculty_name = session['faculty_name'],
        matched      = matched,
        unmatched    = unmatched,
        college      = college,
        branch       = branch,
        semester     = semester,
        div          = div,
        subject      = subject,
        mode         = mode,
        id_type      = id_type
    )


# -------------------------------------------------------------------
# CONFIRM ATTENDANCE — saves to DB after faculty reviews OCR preview
# -------------------------------------------------------------------

@app.route('/confirm_attendance', methods=['POST'])
@login_required
def confirm_attendance():
    pending = session.get('pending')
    if not pending:
        return redirect(url_for('upload'))

    college  = pending['college']
    branch   = pending['branch']
    semester = pending['semester']
    div      = pending['div']
    subject  = pending['subject']
    id_type  = pending['id_type']
    matched  = pending['matched']

    conn   = get_connection()
    cursor = conn.cursor()

    # Get all students in this class
    cursor.execute('''
        SELECT id, name, roll_number, enrollment_number
        FROM students
        WHERE college = ? AND branch = ? AND semester = ? AND div = ?
    ''', (college, branch, int(semester), div))
    all_students = cursor.fetchall()

    # Build present set from matched numbers
    roll_map       = {s['roll_number']: s['id'] for s in all_students}
    enrollment_map = {s['enrollment_number']: s['id'] for s in all_students}

    present_ids = set()
    for num in matched:
        if id_type == 'roll' and num in roll_map:
            present_ids.add(roll_map[num])
        elif id_type == 'enrollment' and num in enrollment_map:
            present_ids.add(enrollment_map[num])

    # Create lecture record
    cursor.execute('''
        INSERT INTO lectures (faculty_id, college, branch, semester, div, subject, date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session['faculty_id'], college, branch, int(semester), div, subject, str(date.today())))
    lecture_id = cursor.lastrowid

    # Insert attendance for ALL students
    records = []
    for s in all_students:
        status = 'PRESENT' if s['id'] in present_ids else 'ABSENT'
        records.append((lecture_id, s['id'], status))

    cursor.executemany('''
        INSERT INTO attendance (lecture_id, student_id, status)
        VALUES (?, ?, ?)
    ''', records)

    conn.commit()
    conn.close()

    # Clear pending data from session
    session.pop('pending', None)

    return redirect(url_for('results', lecture_id=lecture_id))


# -------------------------------------------------------------------
# RESULTS PAGE
# -------------------------------------------------------------------

@app.route('/results/<int:lecture_id>', methods=['GET'])
@login_required
def results(lecture_id):
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM lectures WHERE id = ?", (lecture_id,))
    lecture = cursor.fetchone()

    if not lecture:
        conn.close()
        return redirect(url_for('dashboard'))

    cursor.execute('''
        SELECT s.name, s.roll_number, s.enrollment_number, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.lecture_id = ?
        ORDER BY CAST(s.roll_number AS INTEGER)
    ''', (lecture_id,))
    records = cursor.fetchall()
    conn.close()

    present = [r for r in records if r['status'] == 'PRESENT']
    absent  = [r for r in records if r['status'] == 'ABSENT']
    total   = len(records)
    percentage = round((len(present) / total * 100), 1) if total > 0 else 0

    return render_template('results.html',
        faculty_name = session['faculty_name'],
        lecture      = lecture,
        present      = present,
        absent       = absent,
        total        = total,
        percentage   = percentage
    )


# -------------------------------------------------------------------
# STUDENTS PAGE
# -------------------------------------------------------------------

@app.route('/students', methods=['GET'])
@login_required
def students():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT name, roll_number, enrollment_number, branch, semester, div
        FROM students
        WHERE college = ?
        ORDER BY branch, semester, div, CAST(roll_number AS INTEGER)
    ''', (session['college'],))
    all_students = cursor.fetchall()
    conn.close()

    return render_template('students.html',
        faculty_name = session['faculty_name'],
        students     = all_students,
        total        = len(all_students)
    )


# -------------------------------------------------------------------
# REPORTS PAGE
# -------------------------------------------------------------------

@app.route('/reports', methods=['GET'])
@login_required
def reports():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT l.id, l.date, l.subject, l.branch, l.semester, l.div,
               COUNT(CASE WHEN a.status = 'PRESENT' THEN 1 END) as present_count,
               COUNT(a.id) as total_count
        FROM lectures l
        JOIN attendance a ON a.lecture_id = l.id
        WHERE l.faculty_id = ?
        GROUP BY l.id
        ORDER BY l.date DESC
    ''', (session['faculty_id'],))
    lectures = cursor.fetchall()
    conn.close()

    return render_template('reports.html',
        faculty_name = session['faculty_name'],
        lectures     = lectures
    )


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True)