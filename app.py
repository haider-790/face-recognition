from flask import Flask, render_template, request, jsonify
from database import db, Student, Attendance
from face_engine import capture_faces, save_face_capture, train_model, run_attendance_session
from liveness import check_liveness
import os
import pickle
from datetime import datetime

app = Flask(__name__)

instance_dir = os.path.join(app.root_path, 'instance')
os.makedirs(instance_dir, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_dir, 'attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()


# ── DASHBOARD ──────────────────────────────────────────────────
@app.route('/')
def index():
    total_students = Student.query.count()
    total_records  = Attendance.query.count()
    today          = datetime.now().strftime('%Y-%m-%d')
    today_count    = Attendance.query.filter_by(date=today).count()

    # Check trained students count
    labels_path = os.path.join(app.root_path, 'TrainingImageLabel', 'labels.pkl')
    trained_students = 0
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'rb') as f:
                id_map = pickle.load(f)
                trained_students = len(id_map)
        except:
            trained_students = 0

    return render_template('index.html',
                           total_students=total_students,
                           total_records=total_records,
                           today_count=today_count,
                           trained_students=trained_students)


# ── REGISTER STUDENT ───────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        sid  = request.form.get('student_id', '').strip()
        name = request.form.get('name', '').strip()

        if not sid or not name:
            return jsonify({'status': 'error', 'msg': 'Both fields are required.'})

        if Student.query.filter_by(student_id=sid).first():
            return jsonify({'status': 'error', 'msg': f"ID '{sid}' already registered."})

        db.session.add(Student(student_id=sid, name=name))
        db.session.commit()

        return jsonify({'status': 'ok',
                        'msg': f"Registered {name}! Start capturing face images now."})

    return render_template('register.html')


@app.route('/capture-face', methods=['POST'])
def capture_face():
    sid = request.form.get('student_id', '').strip()
    name = request.form.get('name', '').strip()
    image_data = request.form.get('image', '').strip()

    if not sid or not name or not image_data:
        return jsonify({'status': 'error', 'msg': 'Missing student ID, name, or image data.'})

    saved_count = save_face_capture(sid, name, image_data)
    if saved_count is None:
        return jsonify({'status': 'error', 'msg': 'Invalid image data.'})
    elif saved_count == 0:
        return jsonify({'status': 'ignored', 'msg': 'No face detected in this frame.', 'saved_count': 0})

    return jsonify({'status': 'ok', 'msg': 'Face image captured.', 'saved_count': saved_count})


# ── TRAIN MODEL ────────────────────────────────────────────────
@app.route('/train', methods=['POST'])
def train():
    success, msg = train_model()
    return jsonify({'status': 'ok' if success else 'error', 'msg': msg})


# ── MARK ATTENDANCE ────────────────────────────────────────────
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        # Subject is OPTIONAL — leave blank if not needed
        subject = request.form.get('subject', '').strip() or 'General'

        model_path = os.path.join(app.root_path, 'TrainingImageLabel', 'face_model.yml')
        if not os.path.exists(model_path):
            return jsonify({'status': 'error',
                            'msg': 'Model not trained. Go to Dashboard → Train Model first.'})

        # Run recognition session — camera opens, shows names on screen
        recognized = run_attendance_session(subject=subject)

        # Save each recognized student to database
        today = datetime.now().strftime('%Y-%m-%d')
        now   = datetime.now().strftime('%H:%M:%S')
        saved = []

        for person in recognized:
            sid  = person['id']
            name = person['name']
            # Avoid duplicate for same student + subject + date
            exists = Attendance.query.filter_by(
                student_id=sid, subject=subject, date=today
            ).first()
            if not exists:
                db.session.add(Attendance(
                    student_id=sid, name=name,
                    subject=subject, date=today, time=now
                ))
                db.session.commit()
            saved.append(name)

        if saved:
            msg = f"Attendance marked for: {', '.join(saved)}"
        else:
            msg = "No students were recognized. Make sure model is trained and face is visible."

        return jsonify({'status': 'ok', 'marked': saved, 'msg': msg})

    return render_template('attendance.html')


# ── VIEW RECORDS ───────────────────────────────────────────────
@app.route('/view')
def view():
    # Filter by date and/or subject
    sel_date    = request.args.get('date', '')
    sel_subject = request.args.get('subject', '')

    query = Attendance.query
    if sel_date:    query = query.filter_by(date=sel_date)
    if sel_subject: query = query.filter_by(subject=sel_subject)

    records  = query.order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    subjects = [s[0] for s in db.session.query(Attendance.subject).distinct().all()]

    # Summary: how many unique students on selected date
    if sel_date:
        present_count = len(set(r.student_id for r in records))
    else:
        present_count = None

    return render_template('view.html',
                           records=records,
                           subjects=subjects,
                           sel_date=sel_date,
                           sel_subject=sel_subject,
                           present_count=present_count,
                           total_students=Student.query.count())


# ── STUDENTS LIST ──────────────────────────────────────────────
@app.route('/students')
def students():
    return render_template('students.html',
                           students=Student.query.order_by(Student.name).all())


if __name__ == '__main__':
    app.run(debug=True)