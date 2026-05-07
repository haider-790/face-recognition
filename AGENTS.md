# Attendance Management System - AI Agent Instructions

## Project Overview
This is a Flask-based web application for attendance tracking using face recognition. It captures student faces, trains an LBPH model, and performs real-time recognition during attendance sessions. Uses OpenCV for face detection, dlib for optional liveness checks, and SQLite for data storage.

## Key Technologies
- **Backend**: Flask web framework with SQLAlchemy ORM
- **Face Recognition**: OpenCV LBPH algorithm with Haar Cascade detection
- **Database**: SQLite (auto-created in `instance/attendance.db`)
- **Dependencies**: numpy, pandas, pillow, dlib (pre-bundled wheel for Python 3.10/Windows)

## Build and Run Commands
- Install: `pip install -r requirements.txt`
- Run: `python app.py` (serves on http://localhost:5000)
- Train model: POST to `/train` endpoint after registering students
- Attendance session: POST to `/attendance` with subject parameter

## Architecture
- **app.py**: Flask routes for web interface
- **face_engine.py**: Face capture, training, and recognition pipeline
- **database.py**: SQLAlchemy models for Student and Attendance
- **liveness.py**: Optional blink-based liveness detection (requires shape_predictor_68_face_landmarks.dat)
- **templates/**: Jinja2 HTML templates with base.html inheritance
- **static/**: CSS styling with custom properties

## Project Conventions
- Face images stored in `TrainingImage/{student_id}_{name}/` (60 images per student)
- Model saved as `TrainingImageLabel/face_model.yml` with labels.pkl mapping
- Image preprocessing: 150x150 resize, histogram equalization, Gaussian blur
- Recognition confidence threshold: 85% (lower = better match)
- Single attendance mark per student per session
- Windows camera access uses `cv2.CAP_DSHOW`

## Common Pitfalls
- dlib requires C++ compiler; use bundled .whl for Python 3.10/Windows
- Camera access fails silently; test with `cv2.VideoCapture(0, cv2.CAP_DSHOW).isOpened()`
- Paths are hardcoded relative; run from app/ directory
- Training slow with many students; consider background processing
- Liveness check disabled by default; needs .dat file download
- No authentication; web UI publicly accessible

## Key Files for Reference
- [app.py](app.py): Main application and routes
- [face_engine.py](face_engine.py): Face recognition implementation
- [database.py](database.py): Database models and operations
- [templates/view.html](templates/view.html): Attendance viewing with filtering
- [static/style.css](static/style.css): Styling patterns

## Development Notes
- Register students first, then train model before attendance sessions
- Model training blocks UI; consider async for production
- Face detection sensitive to lighting; ensure consistent conditions
- Database prevents duplicates within same date/subject/session</content>
<parameter name="filePath">c:\Users\Haider Ali\Documents\Attendance-Management-system-using-face-recognition\app\AGENTS.md