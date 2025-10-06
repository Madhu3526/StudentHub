import streamlit as st
import sqlite3
import ast
import operator as op
from datetime import datetime, date
import json
import plotly.express as px
import pandas as pd
import numpy as np
import io
from fpdf import FPDF
import pytesseract
from PIL import Image
import re
try:
    import cv2
except ImportError:
    cv2 = None

DB_PATH = "student_kbs.db"

# JSON serialization helper
def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

# ---------------- Safe evaluator ----------------
CMP_OPS = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

def safe_eval(expr, context):
    try:
        node = ast.parse(expr, mode='eval')
        return _eval(node.body, context)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

def _eval(node, context):
    # boolean ops
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, context) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval(v, context) for v in node.values)
        raise ValueError("Unsupported boolean op")
    # comparisons
    if isinstance(node, ast.Compare):
        left = _eval(node.left, context)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, context)
            if type(op_node) not in CMP_OPS:
                raise ValueError(f"Operator {type(op_node)} not allowed")
            func = CMP_OPS[type(op_node)]
            if not func(left, right):
                return False
            left = right
        return True
    # name
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    # literal
    if isinstance(node, ast.Constant):
        return node.value
    # list
    if isinstance(node, ast.List):
        return [_eval(e, context) for e in node.elts]
    # subscript
    if isinstance(node, ast.Subscript):
        value = _eval(node.value, context)
        if hasattr(node.slice, 'value'):
            idx = _eval(node.slice.value, context)
        else:
            idx = _eval(node.slice, context)
        return value[idx]
    raise ValueError(f"Unsupported expression node: {type(node)}")

# ---------------- DB helpers ----------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn(); cur = conn.cursor()

    # users
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE, password TEXT, role TEXT
        )
    ''')

    # students
    cur.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT UNIQUE,
            name TEXT,
            department TEXT,
            gpa REAL DEFAULT 0,
            attendance REAL DEFAULT 0,
            created_at TEXT
        )
    ''')

    # attendance records -> stores per-class attendance
    cur.execute('''
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT,
            status TEXT CHECK(status IN ('present','absent')),
            source TEXT DEFAULT 'manual',
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')

    # activities
    cur.execute('''
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            type TEXT,
            title TEXT,
            date TEXT,
            description TEXT,
            evidence_url TEXT,
            status TEXT DEFAULT 'pending',
            approver_id INTEGER,
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')

    # rules
    cur.execute('''
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            condition TEXT,
            action TEXT,
            priority INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            created_at TEXT
        )
    ''')

    # inferences
    cur.execute('''
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            rule_id INTEGER,
            action_result TEXT,
            evidence TEXT,
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(rule_id) REFERENCES rules(id)
        )
    ''')

    # courses (for academic grading)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sem INTEGER,
            code TEXT,
            name TEXT,
            credits INTEGER DEFAULT 3
        )
    ''')

    # marks (faculty-entered)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            course_id INTEGER,
            mid INTEGER,
            internal INTEGER,
            semester_exam INTEGER,
            final_marks REAL,
            grade TEXT,
            grade_point REAL,
            updated_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(course_id) REFERENCES courses(id),
            UNIQUE(student_id, course_id)
        )
    ''')

    # gpa table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS gpa (
            student_id INTEGER,
            semester INTEGER,
            gpa REAL,
            PRIMARY KEY(student_id, semester),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')

    # certificates
    cur.execute('''
        CREATE TABLE IF NOT EXISTS certificates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            filename TEXT,
            ocr_text TEXT,
            parsed_json TEXT,
            verification_score INTEGER,
            flags TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')

    conn.commit()

    # seed demo users
    try:
        cur.execute("INSERT INTO users(username,password,role) VALUES (?,?,?)", ("admin","admin123","admin"))
        cur.execute("INSERT INTO users(username,password,role) VALUES (?,?,?)", ("user","user123","student"))
    except sqlite3.IntegrityError:
        pass

    conn.commit()
    conn.close()

# ---------------- CRUD ----------------
def add_student(reg_no, name, department, attendance=0.0):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO students(reg_no,name,department,gpa,attendance,created_at) VALUES (?,?,?,?,?,?)",
                (reg_no, name, department, 0.0, attendance, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def list_students():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT * FROM students ORDER BY id DESC")
    rows = cur.fetchall(); conn.close()
    return [dict(r) for r in rows]

def get_student(student_id):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE id=?", (student_id,))
    row = cur.fetchone(); conn.close()
    return row

def get_student_by_reg(reg_no):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE reg_no=?", (reg_no,))
    row = cur.fetchone(); conn.close()
    return row

def add_rule(name, condition, action, priority=0, enabled=1):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO rules(name,condition,action,priority,enabled,created_at) VALUES(?,?,?,?,?,?)",
                (name, condition, action, priority, enabled, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def list_rules():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT * FROM rules WHERE enabled=1 ORDER BY priority DESC")
    rows = cur.fetchall(); conn.close()
    return rows

def add_activity(student_id, type_, title, date, description, evidence_url):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO activities(student_id,type,title,date,description,evidence_url,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (student_id, type_, title, date, description, evidence_url, 'pending', datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def verify_activity_certificate_match(activity_title, student_name, parsed_cert):
    """Check if certificate matches the activity details"""
    match_score = 0
    flags = []
    
    # Check name match
    if parsed_cert.get('name'):
        if student_name.lower() in parsed_cert['name'].lower():
            match_score += 30
        else:
            flags.append("Student name mismatch")
    
    # Check course/title match
    if parsed_cert.get('course') and activity_title:
        activity_words = activity_title.lower().split()
        cert_course = parsed_cert['course'].lower()
        if any(word in cert_course for word in activity_words if len(word) > 3):
            match_score += 40
        else:
            flags.append("Certificate course doesn't match activity")
    
    return match_score, flags

def list_pending_activities():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT a.*, s.name, s.reg_no FROM activities a JOIN students s ON a.student_id=s.id WHERE a.status='pending' ORDER BY a.created_at DESC")
    rows = cur.fetchall(); conn.close()
    return rows

def approve_activity(activity_id, approver_id, approve=True):
    conn = get_conn(); cur = conn.cursor()
    status = 'approved' if approve else 'rejected'
    cur.execute("UPDATE activities SET status=?, approver_id=? WHERE id=?", (status, approver_id, activity_id))
    conn.commit(); conn.close()

def store_inference(student_id, rule_id, action_result, evidence):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO inferences(student_id,rule_id,action_result,evidence,created_at) VALUES(?,?,?,?,?)",
                (student_id, rule_id, action_result, json.dumps(evidence), datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

# ---------------- Certificate Processing ----------------
def process_certificate(image_bytes, student_id, filename):
    """Unified certificate processing: OCR + parsing + verification + DB storage"""
    try:
        # Enhanced OCR with preprocessing
        image = Image.open(io.BytesIO(image_bytes))
        
        # Image preprocessing for better OCR
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply image enhancements
        img_array = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img_array = cv2.GaussianBlur(img_array, (1, 1), 0)
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        enhanced_image = Image.fromarray(img_array)
        
        # OCR with custom config
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/\s'
        ocr_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
        
        # Parse certificate using comprehensive regex patterns
        parsed = {}
        text_lower = ocr_text.lower()
        
        # Extract name with multiple patterns
        name_patterns = [
            r'name[:\s]+([a-zA-Z\s.,-]+)',
            r'this is to certify that[\s]+([a-zA-Z\s.,-]+)',
            r'student name[:\s]+([a-zA-Z\s.,-]+)'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                parsed['name'] = match.group(1).strip().title()
                break
        
        # Extract course/certification
        course_patterns = [
            r'course[:\s]+([a-zA-Z\s.,-]+)',
            r'certification in[\s]+([a-zA-Z\s.,-]+)',
            r'completed[\s]+([a-zA-Z\s.,-]+)',
            r'certificate\s+of[\s]+([a-zA-Z\s.,-]+)'
        ]
        for pattern in course_patterns:
            match = re.search(pattern, text_lower)
            if match:
                parsed['course'] = match.group(1).strip().title()
                break
        
        # Extract date
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(\d{1,2}\s+[a-zA-Z]+\s+\d{4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, ocr_text)
            if match:
                parsed['date'] = match.group(1)
                break
        
        # Extract organization
        org_patterns = [
            r'issued by[\s]+([a-zA-Z\s.,-]+)',
            r'from[\s]+([a-zA-Z\s.,-]+)',
            r'by[\s]+([a-zA-Z\s.,-]+)'
        ]
        for pattern in org_patterns:
            match = re.search(pattern, text_lower)
            if match:
                parsed['organization'] = match.group(1).strip().title()
                break
        
        # Extract certificate ID
        cert_id_patterns = [
            r'certificate\s*(id|no)[:\s]+([a-zA-Z0-9\-\/]+)',
            r'id[:\s]+([a-zA-Z0-9\-\/]+)'
        ]
        for pattern in cert_id_patterns:
            match = re.search(pattern, text_lower)
            if match:
                parsed['cert_id'] = match.group(2).strip().upper()
                break
        
        # Verification scoring
        score = 0
        flags = []
        
        # Field-based scoring
        if parsed.get('name'):
            score += 25
        else:
            flags.append("Missing name")
        
        if parsed.get('course'):
            score += 25
        else:
            flags.append("Missing course/certification")
        
        if parsed.get('date'):
            score += 20
        else:
            flags.append("Missing date")
        
        if parsed.get('organization'):
            score += 20
        else:
            flags.append("Missing organization")
        
        # Keyword verification
        cert_keywords = ['certificate', 'certification', 'completion', 'achievement', 'awarded']
        if any(keyword in text_lower for keyword in cert_keywords):
            score += 10
        else:
            flags.append("No certificate keywords found")
        
        # Enhanced student name verification with fuzzy matching
        student = get_student(student_id)
        if student and parsed.get('name'):
            student_name = re.sub(r'[^a-zA-Z\s]', '', student['name'].lower().strip())
            cert_name = re.sub(r'[^a-zA-Z\s]', '', parsed['name'].lower().strip())
            
            # Multiple matching strategies
            name_match = False
            
            # Exact match
            if student_name in cert_name or cert_name in student_name:
                name_match = True
            
            # Word-by-word matching
            if not name_match:
                student_words = set(student_name.split())
                cert_words = set(cert_name.split())
                common_words = student_words.intersection(cert_words)
                if len(common_words) >= min(2, len(student_words)):
                    name_match = True
            
            # Initials matching
            if not name_match:
                student_initials = ''.join([word[0] for word in student_name.split() if word])
                if student_initials in cert_name.replace(' ', ''):
                    name_match = True
            
            if name_match:
                score += 20
            else:
                flags.append(f"Name mismatch: '{student['name']}' vs '{parsed['name']}'")
                score = max(0, score - 10)  # Reduced penalty
        
        final_score = min(score, 100)
        
        # Store in database
        conn = get_conn(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO certificates(student_id, filename, ocr_text, parsed_json, verification_score, flags, status, created_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (student_id, filename, ocr_text, json.dumps(parsed), final_score, json.dumps(flags), "pending", datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
        
        return ocr_text, parsed, final_score, flags
        
    except Exception as e:
        return str(e), {}, 0, ["Error processing certificate"]

def list_pending_activities():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT a.*, s.name, s.reg_no FROM activities a JOIN students s ON a.student_id=s.id WHERE a.status='pending' ORDER BY a.created_at DESC")
    rows = cur.fetchall(); conn.close()
    return rows

def approve_activity(activity_id, approver_id, approve=True):
    conn = get_conn(); cur = conn.cursor()
    status = 'approved' if approve else 'rejected'
    cur.execute("UPDATE activities SET status=?, approver_id=? WHERE id=?", (status, approver_id, activity_id))
    conn.commit(); conn.close()

def store_inference(student_id, rule_id, action_result, evidence):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO inferences(student_id,rule_id,action_result,evidence,created_at) VALUES(?,?,?,?,?)",
                (student_id, rule_id, action_result, json.dumps(evidence), datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

# ---------------- Attendance helpers ----------------
def add_attendance_record(student_id, dt, status, source='manual'):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO attendance_records(student_id,date,status,source,created_at) VALUES(?,?,?,?,?)",
                (student_id, dt.isoformat() if isinstance(dt, date) else dt, status, source, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def calculate_attendance(student_id, start_date=None, end_date=None):
    conn = get_conn()
    q = "SELECT date, status FROM attendance_records WHERE student_id=?"
    params = [student_id]
    if start_date and end_date:
        q += " AND date BETWEEN ? AND ?"
        params += [start_date.isoformat(), end_date.isoformat()]
    df = pd.read_sql_query(q, conn, params=tuple(params))
    conn.close()
    if df.empty:
        return None
    df['status_norm'] = df['status'].str.lower().map(lambda x: 'present' if x=='present' else 'absent')
    total = len(df)
    present = (df['status_norm']=='present').sum()
    pct = round((present / total) * 100.0, 2)
    conn = get_conn(); cur = conn.cursor()
    cur.execute("UPDATE students SET attendance=? WHERE id=?", (pct, student_id))
    conn.commit(); conn.close()
    return pct

# ---------------- Inference & context ----------------
def build_context_for_student(student_row):
    ctx = {}
    if student_row is None:
        return ctx
    calculated_cgpa = calculate_cgpa(student_row['id'])
    ctx['GPA'] = calculated_cgpa if calculated_cgpa is not None else 0
    ctx['Attendance'] = student_row['attendance'] if student_row['attendance'] is not None else 0
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT type, COUNT(*) as cnt FROM activities WHERE student_id=? AND status='approved' GROUP BY type", (student_row['id'],))
    rows = cur.fetchall()
    activity_counts = {r['type']: r['cnt'] for r in rows}
    ctx['ActivityCount'] = activity_counts
    cur.execute("SELECT title FROM activities WHERE student_id=? AND status='approved' AND type='certification'", (student_row['id'],))
    certs = [r['title'] for r in cur.fetchall()]
    ctx['Certifications'] = certs
    ctx['CertificationCount'] = len(certs)
    ctx['ClubActivities'] = activity_counts.get('club', 0)
    ctx['Internships'] = activity_counts.get('internship', 0)
    ctx['Workshops'] = activity_counts.get('workshop', 0)
    ctx['Conferences'] = activity_counts.get('conference', 0)
    conn.close()
    return ctx

def evaluate_rules_for_student(student_id):
    student = get_student(student_id)
    if not student:
        return []
    ctx = build_context_for_student(student)
    results = []
    rules = list_rules()
    for r in rules:
        try:
            ok = safe_eval(r['condition'], ctx)
        except Exception:
            ok = False
        if ok:
            evidence = {'student': dict(student), 'context': ctx, 'rule_condition': r['condition']}
            store_inference(student_id, r['id'], r['action'], evidence)
            results.append({'rule': r['name'], 'action': r['action'], 'evidence': evidence})
    return results

# ---------------- Academic grading helpers ----------------
GRADE_SCALE = [
    (90, "O", 10),
    (80, "A+", 9),
    (70, "A", 8),
    (60, "B+", 7),
    (50, "B", 6),
    (40, "C", 5),
    (0, "F", 0),
]

def calculate_final_marks(mid, internal, semester_exam):
    combined = (mid if mid is not None else 0) + (internal if internal is not None else 0)
    scaled_mid_internal = (combined / 100.0) * 40.0
    scaled_sem = (semester_exam / 100.0) * 60.0 if semester_exam is not None else 0.0
    final = scaled_mid_internal + scaled_sem
    return round(final, 2)

def get_grade_and_gp(final_marks):
    for cutoff, grade, gp in GRADE_SCALE:
        if final_marks >= cutoff:
            return grade, gp
    return "F", 0

def add_or_update_marks(student_id, course_id, mid, internal, semester_exam):
    final = calculate_final_marks(mid, internal, semester_exam)
    grade, gp = get_grade_and_gp(final)
    conn = get_conn(); cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO marks(student_id, course_id, mid, internal, semester_exam, final_marks, grade, grade_point, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(student_id, course_id) DO UPDATE SET
          mid=excluded.mid,
          internal=excluded.internal,
          semester_exam=excluded.semester_exam,
          final_marks=excluded.final_marks,
          grade=excluded.grade,
          grade_point=excluded.grade_point,
          updated_at=excluded.updated_at
    """, (student_id, course_id, mid, internal, semester_exam, final, grade, gp, now))
    conn.commit(); conn.close()
    return final, grade, gp

def calculate_gpa(student_id, semester):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT m.final_marks, m.grade_point, c.credits
        FROM marks m JOIN courses c ON m.course_id=c.id
        WHERE m.student_id=? AND c.sem=?
    """, conn, params=(student_id, semester))
    conn.close()
    if df.empty:
        return None
    total_points = (df['grade_point'] * df['credits']).sum()
    total_credits = df['credits'].sum()
    if total_credits == 0:
        return None
    gpa_val = round(total_points / total_credits, 2)
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO gpa(student_id, semester, gpa) VALUES (?,?,?)", (student_id, semester, gpa_val))
    conn.commit(); conn.close()
    return gpa_val

def calculate_cgpa(student_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT gpa FROM gpa WHERE student_id=?", conn, params=(student_id,))
    conn.close()
    if df.empty:
        return None
    return round(df['gpa'].mean(), 2)

# ---------------- PDF Resume generator ----------------
def safe_text(s):
    if s is None:
        return ""
    return str(s).encode("latin-1", "replace").decode("latin-1")

def generate_resume_pdf_bytes(student_id):
    student = get_student(student_id)
    if not student:
        raise ValueError("Student not found")
    cgpa = calculate_cgpa(student_id) or 0.0
    attendance = student['attendance'] or 0.0
    conn = get_conn()
    df_act = pd.read_sql_query("SELECT * FROM activities WHERE student_id=? AND status='approved' ORDER BY date DESC", conn, params=(student_id,))
    df_marks = pd.read_sql_query("""SELECT c.sem, c.code, c.name, c.credits, m.final_marks, m.grade
           FROM marks m JOIN courses c ON m.course_id=c.id
           WHERE m.student_id=? ORDER BY c.sem, c.code""", conn, params=(student_id,))
    conn.close()
    ctx = build_context_for_student(student)
    badges = []
    if cgpa >= 8.5: badges.append("Academic Star")
    if ctx.get('ClubActivities', 0) >= 2: badges.append("Club Champion")
    if ctx.get('Internships', 0) >= 1: badges.append("Industry Intern")
    if ctx.get('CertificationCount', 0) >= 2: badges.append("Certified")
    
    # Calculate activity credits
    credit_mapping = {
        'conference': 2, 'workshop': 1, 'certification': 3, 'internship': 5, 'club': 1, 'project': 4, 'sports': 2,
        'leadership': 3, 'community_service': 2, 'competition': 3, 'research': 5, 'award': 4, 'volunteering': 1
    }
    total_credits = sum(credit_mapping.get(row['type'], 0) for _, row in df_act.iterrows())
    
    pdf = FPDF()
    pdf.add_page()
    
    # Header with name and title
    pdf.set_fill_color(41, 128, 185)  # Blue background
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", 'B', 20)
    pdf.ln(8)
    pdf.cell(0, 12, safe_text(student['name'].upper()), ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, safe_text(f"Student Portfolio | {student['department']}"), ln=True, align="C")
    
    # Reset colors
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)
    
    # Personal Information Section
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, safe_text("PERSONAL INFORMATION"), ln=True, align="L")
    pdf.set_font("Arial", size=11)
    pdf.cell(95, 6, safe_text(f"Registration Number: {student['reg_no']}"), 0, 0)
    pdf.cell(95, 6, safe_text(f"CGPA: {cgpa:.2f}/10.0"), ln=True)
    pdf.cell(95, 6, safe_text(f"Department: {student['department']}"), 0, 0)
    pdf.cell(95, 6, safe_text(f"Attendance: {attendance:.1f}%"), ln=True)
    pdf.ln(8)
    
    # Achievements & Badges
    if badges:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, safe_text("ACHIEVEMENTS & BADGES"), ln=True)
        pdf.set_font("Arial", size=11)
        for badge in badges:
            pdf.cell(0, 6, safe_text(f"• {badge}"), ln=True)
        pdf.ln(4)
    
    # Activity Credits Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, safe_text("ACTIVITY SUMMARY"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, safe_text(f"Total Activity Credits Earned: {total_credits}"), ln=True)
    pdf.cell(0, 6, safe_text(f"Total Approved Activities: {len(df_act)}"), ln=True)
    pdf.ln(4)
    
    # Academic Performance
    if not df_marks.empty:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, safe_text("ACADEMIC PERFORMANCE"), ln=True)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(20, 6, "Sem", 1, 0, 'C')
        pdf.cell(25, 6, "Code", 1, 0, 'C')
        pdf.cell(80, 6, "Course Name", 1, 0, 'C')
        pdf.cell(20, 6, "Credits", 1, 0, 'C')
        pdf.cell(20, 6, "Marks", 1, 0, 'C')
        pdf.cell(15, 6, "Grade", 1, 1, 'C')
        
        pdf.set_font("Arial", size=9)
        for _, r in df_marks.head(15).iterrows():  # Limit to prevent overflow
            pdf.cell(20, 5, str(r['sem']), 1, 0, 'C')
            pdf.cell(25, 5, safe_text(r['code']), 1, 0, 'C')
            pdf.cell(80, 5, safe_text(r['name'][:35]), 1, 0, 'L')  # Truncate long names
            pdf.cell(20, 5, str(r['credits']), 1, 0, 'C')
            pdf.cell(20, 5, str(r['final_marks']), 1, 0, 'C')
            pdf.cell(15, 5, safe_text(r['grade']), 1, 1, 'C')
        pdf.ln(4)
    
    # Extracurricular Activities
    if not df_act.empty:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, safe_text("EXTRACURRICULAR ACTIVITIES"), ln=True)
        
        # Group activities by type
        activity_groups = df_act.groupby('type')
        for activity_type, group in activity_groups:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 6, safe_text(f"{activity_type.upper()} ({len(group)} activities)"), ln=True)
            pdf.set_font("Arial", size=10)
            
            for _, a in group.head(5).iterrows():  # Limit per category
                pdf.cell(0, 5, safe_text(f"• {a['title']} ({a['date']})"), ln=True)
                if len(safe_text(a['description'])) > 0:
                    pdf.set_font("Arial", size=9)
                    pdf.multi_cell(0, 4, safe_text(f"  {a['description'][:100]}..."))
                    pdf.set_font("Arial", size=10)
            pdf.ln(2)
    
    # Footer
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, safe_text(f"Generated on {datetime.now().strftime('%B %d, %Y')} by Smart Student Hub\nThis is a verified digital portfolio showcasing academic and extracurricular achievements."))
    
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1", "replace")
    return pdf_bytes



# ---------------- UI ----------------
init_db()
st.set_page_config(
    page_title="Smart Student Hub", 
    layout='wide',
    page_icon="🎓",
    initial_sidebar_state="expanded"
)
st.title("🎓 Smart Student Hub")
st.markdown("*Centralized Student Achievement Management Platform*")

# Sidebar with additional info
st.sidebar.markdown("### 🎓 Smart Student Hub")
st.sidebar.markdown("*Centralized Student Achievement Management*")
st.sidebar.markdown("---")

menu = st.sidebar.selectbox("Mode", ["Home", "Admin", "Student"])

# Home dashboard
if menu == "Home":
    st.subheader("🏫 Smart Student Hub - Institutional Dashboard")
    
    conn = get_conn()
    total_students = pd.read_sql_query("SELECT COUNT(*) as c FROM students", conn)['c'][0]
    approved_acts = pd.read_sql_query("SELECT COUNT(*) as c FROM activities WHERE status='approved'", conn)['c'][0]
    total_courses = pd.read_sql_query("SELECT COUNT(*) as c FROM courses", conn)['c'][0]
    pending_approvals = pd.read_sql_query("SELECT COUNT(*) as c FROM activities WHERE status='pending'", conn)['c'][0]
    total_certificates = pd.read_sql_query("SELECT COUNT(*) as c FROM certificates WHERE status='approved'", conn)['c'][0]
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Students", int(total_students))
    col2.metric("✅ Approved Activities", int(approved_acts))
    col3.metric("📚 Courses", int(total_courses))
    col4.metric("⏳ Pending Approvals", int(pending_approvals))
    col5.metric("🏆 Verified Certificates", int(total_certificates))
    
    # Quick institutional insights
    st.markdown("---")
    st.subheader("📊 Quick Insights")
    
    # Active students percentage
    active_students = pd.read_sql_query("SELECT COUNT(DISTINCT student_id) as c FROM activities WHERE status='approved'", conn)['c'][0]
    if total_students > 0:
        participation_rate = (active_students / total_students) * 100
        st.metric("Student Participation Rate", f"{participation_rate:.1f}%")
    
    conn.close()
    conn = get_conn()
    df_all = pd.read_sql_query("SELECT a.*, s.name as student_name, s.reg_no FROM activities a LEFT JOIN students s ON a.student_id=s.id WHERE a.status='approved'", conn)
    conn.close()
    if not df_all.empty:
        fig1 = px.pie(df_all, names="type", title="Activities Distribution (approved)")
        st.plotly_chart(fig1, use_container_width=True)
        df_all["date"] = pd.to_datetime(df_all["date"], errors='coerce')
        df_all = df_all.dropna(subset=["date"])
        if not df_all.empty:
            df_all["month"] = df_all["date"].dt.to_period("M").astype(str)
            trends = df_all.groupby("month").size().reset_index(name="count")
            fig2 = px.line(trends, x="month", y="count", title="Activity Submissions Over Time")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No approved activity data yet.")

# Admin portal
if menu == "Admin":
    st.header("Admin Portal (demo)")
    with st.expander("Admin Login (demo)"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", value="admin123", type="password")
        if st.button("Login as Admin"):
            if username == 'admin' and password == 'admin123':
                st.session_state['is_admin'] = True
                st.success("Authenticated as admin")
            else:
                st.error("Invalid credentials")

    if st.session_state.get('is_admin'):
        tabs = st.tabs(["Students","Rules","Approvals","Inferences","Courses & Marks","Attendance","Run Inference","Leaderboards","Analytics","Certificates"])
        # Students
        with tabs[0]:
            st.subheader("Add Student")
            c1, c2 = st.columns(2)
            with c1:
                reg_no = st.text_input("Reg No")
                name = st.text_input("Name")
            with c2:
                dept = st.text_input("Department")
                attendance = st.number_input("Attendance (%) (initial)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
            if st.button("Add Student"):
                if not reg_no or not name:
                    st.error("Reg no & name required")
                else:
                    add_student(reg_no, name, dept, attendance)
                    st.success("Student added (attendance will be recalculated from records if added)")
            st.markdown("---")
            rows = list_students()
            if rows:
                df = pd.DataFrame(rows)
                df['calculated_cgpa'] = df['id'].apply(lambda x: calculate_cgpa(x) or 0.0)
                cols = ['id','reg_no','name','department','calculated_cgpa','attendance']
                avail = [c for c in cols if c in df.columns]
                st.dataframe(df[avail])
            else:
                st.info("No students yet.")

        # Rules
        with tabs[1]:
            st.subheader("Create Rule")
            rule_name = st.text_input("Rule Name")
            condition = st.text_area("Condition (variables: GPA, Attendance, Certifications (list), ActivityCount (dict))")
            action = st.text_input("Action (text tag/result)")
            priority = st.number_input("Priority", value=0)
            if st.button("Add Rule"):
                try:
                    test_ctx = {'GPA':0, 'Attendance':0, 'Certifications':[], 'ActivityCount':{}}
                    safe_eval(condition, test_ctx)
                    add_rule(rule_name, condition, action, priority, 1)
                    st.success("Rule added")
                except Exception as e:
                    st.error(f"Invalid condition: {e}")
            st.markdown("---")
            st.subheader("Enabled Rules")
            rules = list_rules()
            for r in rules:
                st.write(f"{r['id']}: {r['name']} — if `{r['condition']}` -> {r['action']} (priority {r['priority']})")

        # Approvals
        with tabs[2]:
            st.subheader("Pending Approvals")
            pend = list_pending_activities()
            if not pend:
                st.info("No pending activities")
            for a in pend:
                st.write(f"#{a['id']} — {a['title']} by {a['name']} ({a['reg_no']}) — {a['type']}")
                st.write(a['description'])
                if a['evidence_url']:
                    st.write("Evidence:", a['evidence_url'])
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"Approve {a['id']}", key=f"ap_{a['id']}"):
                        approve_activity(a['id'], approver_id=1, approve=True); st.success("Approved")
                with c2:
                    if st.button(f"Reject {a['id']}", key=f"rj_{a['id']}"):
                        approve_activity(a['id'], approver_id=1, approve=False); st.info("Rejected")

        # Inferences
        with tabs[3]:
            st.subheader("Inferences Log")
            conn = get_conn(); cur = conn.cursor()
            cur.execute("SELECT i.*, s.reg_no, s.name, r.name as rule_name FROM inferences i JOIN students s ON i.student_id=s.id JOIN rules r ON i.rule_id=r.id ORDER BY i.created_at DESC")
            logs = cur.fetchall(); conn.close()
            if not logs:
                st.info("No inferences yet")
            else:
                for l in logs[:100]:
                    st.write(f"{l['created_at']}: {l['reg_no']} - {l['name']} | Rule: {l['rule_name']} -> {l['action_result']}")

        # Courses & Marks (Academic)
        with tabs[4]:
            st.subheader("Courses Management")
            c1, c2, c3 = st.columns(3)
            with c1:
                sem_in = st.number_input("Semester (add course)", min_value=1, max_value=12, value=1)
                code_in = st.text_input("Course Code (e.g., CS101)")
            with c2:
                name_in = st.text_input("Course Name")
                credits_in = st.number_input("Credits", min_value=1, max_value=6, value=3)
            with c3:
                if st.button("Add Course"):
                    conn = get_conn(); cur = conn.cursor()
                    cur.execute("INSERT INTO courses(sem, code, name, credits) VALUES (?,?,?,?)", (sem_in, code_in, name_in, credits_in))
                    conn.commit(); conn.close(); st.success("Course added")

            st.markdown("---")
            st.subheader("Enter/Update Marks for a Student")
            students = list_students()
            if not students:
                st.info("Add students first")
            else:
                student_map = {f"{s['reg_no']} - {s['name']}": s['id'] for s in students}
                student_sel = st.selectbox("Select Student", list(student_map.keys()))
                if student_sel:
                    sid = student_map[student_sel]
                    sem_for_marks = st.number_input("Select Semester for marks", min_value=1, max_value=12, value=1)
                    conn = get_conn()
                    df_courses = pd.read_sql_query("SELECT * FROM courses WHERE sem=?", conn, params=(sem_for_marks,))
                    conn.close()
                    if df_courses.empty:
                        st.info("No courses for this semester. Add courses above.")
                    else:
                        course_map = {f"{row['code']} - {row['name']} (Credits:{row['credits']})": row['id'] for _, row in df_courses.iterrows()}
                        course_sel = st.selectbox("Select Course", list(course_map.keys()))
                        mid_in = st.number_input("Mid/CAT Marks (out of 60)", min_value=0, max_value=60, value=0)
                        internal_in = st.number_input("Internal Marks (out of 40)", min_value=0, max_value=40, value=0)
                        sem_exam_in = st.number_input("Semester Exam Marks (out of 100)", min_value=0, max_value=100, value=0)
                        if st.button("Save Marks"):
                            c_id = course_map[course_sel]
                            final, grade, gp = add_or_update_marks(sid, c_id, mid_in, internal_in, sem_exam_in)
                            st.success(f"Saved: Final={final} | Grade={grade} | GradePoint={gp}")
                            gpa_val = calculate_gpa(sid, sem_for_marks)
                            if gpa_val is not None:
                                st.info(f"Updated Semester {sem_for_marks} GPA: {gpa_val}")
                                cg = calculate_cgpa(sid)
                                if cg is not None:
                                    st.info(f"Updated CGPA: {cg}")

        # Attendance management
        with tabs[5]:
            st.subheader("Attendance Management")
            students = list_students()
            student_map = {f"{s['reg_no']} - {s['name']}": s['id'] for s in students}
            sel = st.selectbox("Select Student for attendance", list(student_map.keys()) if students else [])
            if sel:
                sid = student_map[sel]
                col1, col2, col3 = st.columns(3)
                with col1:
                    dt = st.date_input("Date")
                    status = st.selectbox("Status", ['present','absent'])
                    if st.button("Add Attendance Record"):
                        add_attendance_record(sid, dt, status)
                        st.success("Attendance record added")
                with col2:
                    if st.button("Recalculate Attendance for Student"):
                        pct = calculate_attendance(sid)
                        if pct is None:
                            st.info("No attendance records found for this student.")
                        else:
                            st.success(f"Attendance recalculated: {pct}%")
                with col3:
                    if st.button("Recalculate All Students' Attendance"):
                        rows = list_students()
                        updated = 0
                        for r in rows:
                            pct = calculate_attendance(r['id'])
                            if pct is not None:
                                updated += 1
                        st.success(f"Recalculated attendance for {updated} students")

                conn = get_conn()
                df_att = pd.read_sql_query("SELECT date, status FROM attendance_records WHERE student_id=? ORDER BY date", conn, params=(sid,))
                conn.close()
                if not df_att.empty:
                    df_att['date'] = pd.to_datetime(df_att['date'], errors='coerce')
                    df_att = df_att.dropna(subset=['date'])
                    df_att['month'] = df_att['date'].dt.to_period("M").astype(str)
                    monthly = df_att.groupby(['month','status']).size().reset_index(name='count')
                    fig = px.bar(monthly, x='month', y='count', color='status', barmode='group', title='Monthly Attendance (present/absent)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No attendance records yet for this student.")

        # Run Inference (admin)
        with tabs[6]:
            st.subheader("Run Inference for Student")
            students = list_students()
            options = {str(r['id']): f"{r['reg_no']} - {r['name']}" for r in students}
            sel = st.selectbox("Select student", options.keys()) if students else None
            if sel and st.button("Evaluate Rules"):
                res = evaluate_rules_for_student(int(sel))
                if res:
                    st.success(f"{len(res)} inferences produced")
                    for item in res:
                        st.write(item['rule'], "->", item['action'])
                        st.json(item['evidence'])
                else:
                    st.info("No rules matched")

        # Leaderboards
        with tabs[7]:
            st.subheader("Leaderboards")
            conn = get_conn()
            df_lb = pd.read_sql_query("""
                SELECT s.id, s.name, s.reg_no, s.gpa, COUNT(a.id) as approved_activities
                FROM students s LEFT JOIN activities a ON s.id=a.student_id AND a.status='approved'
                GROUP BY s.id ORDER BY approved_activities DESC, s.gpa DESC LIMIT 10
            """, conn)
            conn.close()
            if not df_lb.empty:
                st.table(df_lb)
            else:
                st.info("No data yet")

        # Analytics & Reports
        with tabs[8]:
            st.subheader("Institutional Analytics & Reports")
            
            # NAAC Report Generation
            if st.button("Generate NAAC Report"):
                conn = get_conn()
                
                # Student participation statistics
                df_stats = pd.read_sql_query("""
                    SELECT 
                        s.department,
                        COUNT(DISTINCT s.id) as total_students,
                        COUNT(DISTINCT CASE WHEN a.status='approved' THEN a.student_id END) as active_students,
                        COUNT(CASE WHEN a.status='approved' THEN a.id END) as total_activities,
                        AVG(s.attendance) as avg_attendance,
                        AVG(CASE WHEN g.gpa IS NOT NULL THEN g.gpa END) as avg_gpa
                    FROM students s 
                    LEFT JOIN activities a ON s.id=a.student_id
                    LEFT JOIN (SELECT student_id, AVG(gpa) as gpa FROM gpa GROUP BY student_id) g ON s.id=g.student_id
                    GROUP BY s.department
                """, conn)
                
                # Activity type distribution
                df_activities = pd.read_sql_query("""
                    SELECT type, COUNT(*) as count 
                    FROM activities WHERE status='approved' 
                    GROUP BY type ORDER BY count DESC
                """, conn)
                
                conn.close()
                
                st.write("**Department-wise Statistics:**")
                st.dataframe(df_stats)
                
                st.write("**Activity Distribution:**")
                fig = px.bar(df_activities, x='type', y='count', title='Approved Activities by Type')
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate downloadable report
                report_data = {
                    'department_stats': df_stats.to_dict('records'),
                    'activity_distribution': df_activities.to_dict('records'),
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                st.download_button(
                    "Download NAAC Report (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"naac_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            # Credit-based activity tracking
            st.markdown("---")
            st.subheader("Activity Credit System")
            
            credit_mapping = {
                'conference': 2, 'workshop': 1, 'certification': 3,
                'internship': 5, 'club': 1, 'project': 4, 'sports': 2,
                'leadership': 3, 'community_service': 2, 'competition': 3,
                'research': 5, 'award': 4, 'volunteering': 1
            }
            
            conn = get_conn()
            df_credits = pd.read_sql_query("""
                SELECT s.reg_no, s.name, s.department,
                       COUNT(a.id) as total_activities,
                       GROUP_CONCAT(a.type) as activity_types
                FROM students s 
                LEFT JOIN activities a ON s.id=a.student_id AND a.status='approved'
                GROUP BY s.id
            """, conn)
            conn.close()
            
            if not df_credits.empty:
                df_credits['total_credits'] = df_credits['activity_types'].apply(
                    lambda x: sum(credit_mapping.get(t.strip(), 0) for t in (x.split(',') if x else []))
                )
                df_credits['credit_status'] = df_credits['total_credits'].apply(
                    lambda x: 'Excellent' if x >= 20 else 'Good' if x >= 10 else 'Needs Improvement'
                )
                
                st.dataframe(df_credits[['reg_no', 'name', 'department', 'total_activities', 'total_credits', 'credit_status']])
            
        # Certificates (Admin approval)
        with tabs[9]:
            st.subheader("Pending Certificates")
            conn = get_conn()
            df_cert = pd.read_sql_query("""
                SELECT c.*, s.name, s.reg_no FROM certificates c
                JOIN students s ON c.student_id=s.id
                WHERE c.status='pending'
            """, conn); conn.close()
            if df_cert.empty:
                st.info("No pending certificates")
            else:
                for _, row in df_cert.iterrows():
                    st.write(f"#{row['id']} — {row['filename']} (Student: {row['name']} [{row['reg_no']}])")
                    st.write("Score:", row['verification_score'], "Flags:", row['flags'])
                    try:
                        st.json(json.loads(row['parsed_json']))
                    except Exception:
                        st.write("Parsed:", row['parsed_json'])
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"Approve Cert {row['id']}", key=f"apc_{row['id']}"):
                            conn = get_conn(); conn.execute("UPDATE certificates SET status='approved' WHERE id=?", (row['id'],)); conn.commit(); conn.close()
                            st.success("Certificate approved")
                    with c2:
                        if st.button(f"Reject Cert {row['id']}", key=f"rjc_{row['id']}"):
                            conn = get_conn(); conn.execute("UPDATE certificates SET status='rejected' WHERE id=?", (row['id'],)); conn.commit(); conn.close()
                            st.info("Certificate rejected")

# Student portal
if menu == "Student":
    st.header("Student Portal")
    with st.expander("Login"):
        reg_no = st.text_input("Enter Register Number")
        if st.button("Login as Student"):
            student = get_student_by_reg(reg_no)
            if student:
                st.session_state['is_student'] = True
                st.session_state['student_id'] = student['id']
                st.success(f"Welcome {student['name']} ({student['reg_no']})")
            else:
                st.error("Invalid Register Number")

    if st.session_state.get('is_student'):
        sid = st.session_state['student_id']
        student = get_student(sid)
        ctx = build_context_for_student(student)

        st.subheader("My Profile")
        c1, c2 = st.columns([2,3])
        with c1:
            st.write(f"Name: **{student['name']}**")
            st.write(f"Reg No: **{student['reg_no']}**")
            st.write(f"Department: **{student['department']}**")
            calculated_cgpa = calculate_cgpa(sid)
            if calculated_cgpa is not None:
                st.metric("CGPA (Calculated)", f"{calculated_cgpa:.2f}")
            else:
                st.info("No CGPA yet - marks not entered")
            st.metric("Attendance (%)", f"{student['attendance']:.1f}")
        with c2:
            badges = []
            calculated_cgpa = calculate_cgpa(sid) or 0
            if calculated_cgpa >= 8.5: badges.append("Academic Star")
            if ctx.get('ClubActivities',0) >= 2: badges.append("Club Champion")
            if ctx.get('Internships',0) >= 1: badges.append("Industry Intern")
            if ctx.get('CertificationCount',0) >= 2: badges.append("Certified")
            if badges:
                st.write("Badges:")
                for b in badges:
                    st.success(b)
            else:
                st.info("No badges yet")

        st.markdown("---")
        st.subheader("Upload Activity")
        with st.form("act"):
            typ = st.selectbox("Type", ["conference","workshop","certification","internship","club","project","sports","leadership","community_service","competition","research","award","volunteering"])
            title = st.text_input("Title")
            date_in = st.date_input("Date")
            desc = st.text_area("Description")
            evidence = st.text_input("Evidence URL")
            
            # Certificate upload for verification
            st.write("**Upload Certificate**")
            uploaded_cert = st.file_uploader("Upload certificate for this activity", type=["png","jpg","jpeg"], key="activity_cert")
            
            submitted = st.form_submit_button("Submit Activity")
            if submitted:
                activity_id = None
                # First add the activity
                conn = get_conn(); cur = conn.cursor()
                cur.execute("INSERT INTO activities(student_id,type,title,date,description,evidence_url,status,created_at) VALUES(?,?,?,?,?,?,?,?) RETURNING id",
                            (sid, typ, title, date_in.isoformat(), desc, evidence, 'pending', datetime.utcnow().isoformat()))
                activity_id = cur.fetchone()[0]
                conn.commit(); conn.close()
                
                # Process certificate if uploaded
                cert_match_score = 0
                if uploaded_cert:
                    text, parsed, score, flags = process_certificate(uploaded_cert.read(), sid, uploaded_cert.name)
                    
                    # Check if certificate matches activity
                    cert_match_flags = []
                    if parsed.get('name'):
                        if student['name'].lower() in parsed['name'].lower():
                            cert_match_score += 30
                        else:
                            cert_match_flags.append("Name mismatch with student")
                    
                    # Check if certificate title/course matches activity title
                    if parsed.get('course') and title:
                        if any(word in parsed['course'].lower() for word in title.lower().split()):
                            cert_match_score += 40
                        else:
                            cert_match_flags.append("Certificate course doesn't match activity title")
                    
                    # Update certificate with activity matching info
                    conn = get_conn(); cur = conn.cursor()
                    updated_flags = flags + cert_match_flags
                    updated_score = min(score + cert_match_score, 100)
                    cur.execute("UPDATE certificates SET verification_score=?, flags=? WHERE student_id=? AND filename=?",
                                (updated_score, json.dumps(updated_flags), sid, uploaded_cert.name))
                    
                    # Link certificate to activity
                    cur.execute("UPDATE activities SET evidence_url=? WHERE id=?", 
                                (f"Certificate: {uploaded_cert.name} (Score: {updated_score})", activity_id))
                    conn.commit(); conn.close()
                    
                    st.success(f"Activity submitted with certificate! Match score: {cert_match_score}/70")
                    if cert_match_flags:
                        for flag in cert_match_flags:
                            st.warning(flag)
                else:
                    st.success("Activity submitted for approval")

        st.markdown("---")
        st.subheader("Run Rule Inference")
        if st.button("Evaluate Rules for Me"):
            res = evaluate_rules_for_student(sid)
            if res:
                for r in res:
                    st.success(r['action'])
                    st.caption(f"Triggered by {r['rule']}")
            else:
                st.info("No inferences for your profile yet")

        st.markdown("---")
        st.subheader("My Approved Activities & Insights")
        conn = get_conn()
        df_act = pd.read_sql_query("SELECT * FROM activities WHERE student_id=? AND status='approved'", conn, params=(sid,))
        conn.close()
        if not df_act.empty:
            fig = px.pie(df_act, names="type", title="My Activities by Category")
            st.plotly_chart(fig, use_container_width=True)
            df_act["date"] = pd.to_datetime(df_act["date"], errors='coerce')
            df_act = df_act.dropna(subset=["date"])
            if not df_act.empty:
                df_act["month"] = df_act["date"].dt.to_period("M").astype(str)
                trends = df_act.groupby("month").size().reset_index(name="count")
                fig2 = px.bar(trends, x="month", y="count", title="My Activity Timeline")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("You haven’t added any approved activities yet.")

        st.markdown("---")
        st.subheader("My Grades & GPA")
        conn = get_conn()
        df_all_marks = pd.read_sql_query("""
            SELECT c.sem, c.code, c.name, c.credits, m.mid, m.internal, m.semester_exam, m.final_marks, m.grade, m.grade_point
            FROM marks m JOIN courses c ON m.course_id=c.id
            WHERE m.student_id=?
            ORDER BY c.sem, c.code
        """, conn, params=(sid,))
        conn.close()
        if not df_all_marks.empty:
            semesters = sorted(df_all_marks['sem'].unique())
            selected_sem = st.selectbox("Select Semester", semesters)
            
            # Filter marks for selected semester
            df_sem_marks = df_all_marks[df_all_marks['sem'] == selected_sem]
            st.dataframe(df_sem_marks[['code', 'name', 'credits', 'mid', 'internal', 'semester_exam', 'final_marks', 'grade', 'grade_point']])
            
            # Show GPA for selected semester
            gpa_val = calculate_gpa(sid, selected_sem)
            if gpa_val is not None:
                st.metric(f"Semester {selected_sem} GPA", f"{gpa_val:.2f}")
            
            # Show overall CGPA
            cg = calculate_cgpa(sid)
            if cg is not None:
                st.metric("Overall CGPA", f"{cg:.2f}")
        else:
            st.info("No marks available yet. Ask faculty to enter marks.")

        st.markdown("---")
        st.subheader("🤖 Student Chatbot — Ask about your GPA, badges, or activities")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Input handler
        def handle_user_input():
            user_msg = st.session_state.chat_input.strip()
            if user_msg:
                st.session_state.chat_history.append(("You", user_msg))
                response = chatbot_response(sid, user_msg)
                st.session_state.chat_history.append(("KBS Bot", response))
                st.session_state.chat_input = ""

        def chatbot_response(student_id, user_msg):
            user_msg_lower = user_msg.lower()

            # Case 1: GPA Query
            if "gpa" in user_msg_lower:
                sem_number = None
                for word in user_msg_lower.split():
                    if word.isdigit():
                        sem_number = int(word)
                        break
                
                if sem_number:
                    gpa_val = calculate_gpa(student_id, sem_number)
                    if gpa_val is not None:
                        return f"📊 Your GPA for semester {sem_number} is **{gpa_val:.2f}**"
                    else:
                        return f"❌ I couldn't find GPA for semester {sem_number}. Please check if marks are entered."
                else:
                    return "ℹ️ Please specify which semester GPA you want (e.g., 'GPA for sem 4')."

            # Case 2: CGPA Query
            elif "cgpa" in user_msg_lower:
                cgpa = calculate_cgpa(student_id)
                if cgpa is not None:
                    return f"🎯 Your overall CGPA is **{cgpa:.2f}**"
                else:
                    return "❌ No GPA records found to calculate CGPA."

            # Case 3: Rules Engine
            else:
                results = evaluate_rules_for_student(student_id)
                if results:
                    return "🏆 " + " | ".join([r['action'] for r in results])
                else:
                    return "🤔 I couldn't find any relevant academic record for your query. Try asking about 'GPA', 'CGPA', or your academic performance!"

        # Chat input with callback
        st.text_input("Type your message here...", key="chat_input", on_change=handle_user_input)

        # Display chat history
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.chat_message("user").write(msg)
            else:
                st.chat_message("assistant").write(msg)

        # Show my certificates
        st.markdown("---")
        st.subheader("My Certificates")
        conn = get_conn()
        df_certs = pd.read_sql_query("SELECT filename, verification_score, status, created_at FROM certificates WHERE student_id=? ORDER BY created_at DESC", conn, params=(sid,))
        conn.close()
        if not df_certs.empty:
            st.dataframe(df_certs)
        else:
            st.info("No certificates uploaded yet")

        st.markdown("---")
        st.subheader("📊 My Activity Credits & Progress")
        
        # Credit calculation
        credit_mapping = {
            'conference': 2, 'workshop': 1, 'certification': 3,
            'internship': 5, 'club': 1, 'project': 4, 'sports': 2,
            'leadership': 3, 'community_service': 2, 'competition': 3,
            'research': 5, 'award': 4, 'volunteering': 1
        }
        
        conn = get_conn()
        df_my_activities = pd.read_sql_query("SELECT type FROM activities WHERE student_id=? AND status='approved'", conn, params=(sid,))
        conn.close()
        
        if not df_my_activities.empty:
            total_credits = sum(credit_mapping.get(activity_type, 0) for activity_type in df_my_activities['type'])
            required_credits = 20  # Minimum credits for graduation
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Credits Earned", total_credits)
            with col2:
                st.metric("Required Credits", required_credits)
            with col3:
                progress = min(100, (total_credits / required_credits) * 100)
                st.metric("Progress", f"{progress:.1f}%")
            
            # Progress bar
            st.progress(progress / 100)
            
            if total_credits >= required_credits:
                st.success("🎉 Congratulations! You've met the activity credit requirements!")
            else:
                remaining = required_credits - total_credits
                st.info(f"📈 You need {remaining} more credits to meet graduation requirements")
        else:
            st.info("No approved activities yet. Start participating to earn credits!")
        
        st.markdown("---")
        st.subheader("📄 Download Portfolio")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Resume PDF"):
                try:
                    pdf_bytes = generate_resume_pdf_bytes(sid)
                    st.download_button("Download Resume", data=pdf_bytes, file_name=f"resume_{student['reg_no']}.pdf", mime="application/pdf")
                    st.success("Resume generated successfully!")
                except Exception as e:
                    st.error(f"Error generating resume: {e}")
        
        with col2:
            if st.button("Generate Portfolio JSON"):
                try:
                    # Generate comprehensive portfolio data
                    portfolio_data = {
                        'student_info': {
                            'name': student['name'],
                            'reg_no': student['reg_no'],
                            'department': student['department'],
                            'cgpa': calculate_cgpa(sid),
                            'attendance': student['attendance']
                        },
                        'activities': df_act.to_dict('records') if not df_act.empty else [],
                        'certificates': df_certs.to_dict('records') if not df_certs.empty else [],
                        'badges': badges,
                        'total_credits': sum(credit_mapping.get(activity_type, 0) for activity_type in df_my_activities['type']) if not df_my_activities.empty else 0,
                        'generated_at': datetime.utcnow().isoformat()
                    }
                    
                    # Convert to JSON-serializable format
                    portfolio_data = convert_to_json_serializable(portfolio_data)
                    
                    st.download_button(
                        "Download Portfolio",
                        data=json.dumps(portfolio_data, indent=2),
                        file_name=f"portfolio_{student['reg_no']}.json",
                        mime="application/json"
                    )
                    st.success("Portfolio data generated!")
                except Exception as e:
                    st.error(f"Error generating portfolio: {e}")
