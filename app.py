from flask import Flask, render_template, request, jsonify
from database import db, Student, Attendance
from face_engine import capture_faces, save_face_capture, train_model, recognize_face
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


@app.route('/recognize-face', methods=['POST'])
def recognize_from_browser():
    """Recognize a face from browser video frame during attendance"""
    image_data = request.form.get('image', '').strip()
    
    if not image_data:
        return jsonify({'status': 'error', 'msg': 'No image data provided.'})

    result = recognize_face(image_data)
    
    if result is None:
        return jsonify({'status': 'error', 'msg': 'Failed to process image.'})
    elif result == 'unknown':
        return jsonify({'status': 'unknown', 'msg': 'Face not recognized', 'confidence': 0})
    else:
        # result is {id, name, confidence}
        return jsonify({
            'status': 'recognized',
            'student_id': result['id'],
            'student_name': result['name'],
            'confidence': result['confidence']
        })


# ── TRAIN MODEL ────────────────────────────────────────────────
@app.route('/train', methods=['POST'])
def train():
    success, msg = train_model()
    return jsonify({'status': 'ok' if success else 'error', 'msg': msg})


# ── MARK ATTENDANCE ────────────────────────────────────────────
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        subject = request.form.get('subject', '').strip() or 'General'
        marked_students = request.form.getlist('marked_students[]')
        
        if not marked_students:
            return jsonify({'status': 'error', 'msg': 'No students were recognized.'})

        model_path = os.path.join(app.root_path, 'TrainingImageLabel', 'face_model.yml')
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'msg': 'Model not trained. Go to Dashboard → Train Model first.'})

        today = datetime.now().strftime('%Y-%m-%d')
        now   = datetime.now().strftime('%H:%M:%S')
        saved = []
        already_marked = []

        # Parse marked students (format: "id|name")
        for student_str in marked_students:
            parts = student_str.split('|')
            if len(parts) != 2:
                continue
            
            sid, name = parts[0], parts[1]
            
            # Check if attendance already marked for this student today
            exists = Attendance.query.filter_by(student_id=sid, date=today).first()
            if exists:
                already_marked.append(name)
            else:
                db.session.add(Attendance(
                    student_id=sid, name=name,
                    subject=subject, date=today, time=now
                ))
                db.session.commit()
                saved.append(name)

        # Build response message
        msg_parts = []
        if saved:
            msg_parts.append(f"Attendance marked for: {', '.join(saved)}")
        if already_marked:
            msg_parts.append(f"Attendance already marked today for: {', '.join(already_marked)}")

        msg = " | ".join(msg_parts)

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


# ── CLEAR ALL DATA ─────────────────────────────────────────────
@app.route('/clear-data', methods=['POST'])
def clear_data():
    try:
        # Clear database tables
        Attendance.query.delete()
        Student.query.delete()
        db.session.commit()

        # Clear training images
        training_dir = os.path.join(app.root_path, 'TrainingImage')
        if os.path.exists(training_dir):
            import shutil
            shutil.rmtree(training_dir)
            os.makedirs(training_dir, exist_ok=True)

        # Clear trained model files
        model_dir = os.path.join(app.root_path, 'TrainingImageLabel')
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)

        return jsonify({'status': 'ok', 'msg': 'All student data, training images, and trained models have been cleared. System is now fresh.'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'msg': f'Failed to clear data: {str(e)}'})


if __name__ == '__main__':
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open('http://localhost:5000')

    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(debug=True)