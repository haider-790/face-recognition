# Attendance Management System - AI Agent Instructions

## Project Overview
This is a Flask-based web application for attendance tracking using face recognition. It captures student faces, trains an LBPH model, and performs real-time recognition during attendance sessions. Uses OpenCV for face detection, optional dlib-based liveness detection, and SQLite for data storage.

## Key Technologies
- **Backend**: Flask web framework with SQLAlchemy ORM
- **Face Recognition**: OpenCV LBPH algorithm with Haar Cascade detection
- **Database**: SQLite (auto-created in `instance/attendance.db`)
- **Dependencies**: numpy, pandas, pillow, dlib (pre-bundled wheel for Python 3.10/Windows)

## Build and Run Commands
- Install: `pip install -r requirements.txt`
- Run: `python app.py` (serves on http://localhost:5000)
- Register student: visit `/register`
- Train model: automatic after registration or manual trigger if needed
- Attendance session: visit `/attendance`, enter subject, and start session

## Architecture
- **app.py**: Flask routes, registration flow, attendance flow, dashboard data
- **face_engine.py**: Face capture, browser image saving, model training, and live recognition
- **database.py**: SQLAlchemy models for Student and Attendance
- **liveness.py**: Optional blink-based liveness detection (requires shape_predictor_68_face_landmarks.dat)
- **templates/**: Jinja2 HTML templates with modern UI and real-time feedback
- **static/**: Styling, animations, cards, and responsive layout

## Project Conventions
- Face images stored in `TrainingImage/{student_id}_{name}/`
- Model saved in `TrainingImageLabel/face_model.yml`
- Labels map saved in `TrainingImageLabel/labels.pkl`
- Image preprocessing: 150x150 resize, histogram equalization, Gaussian blur
- Recognition threshold updated to **80** for better live matching
- Single attendance mark per student per subject/date
- Attendance session auto-exits after 60 seconds (or 30 seconds with no detections)
- Dashboard counts trained students from `labels.pkl`
- Windows camera access uses `cv2.CAP_DSHOW`

## Supported Functionalities
- Register new students with browser webcam preview
- Save browser-captured face images and preprocess them automatically
- Train LBPH model on all stored student images
- Mark attendance via live OpenCV camera session
- Show trained student count on the dashboard
- Clear all stored students and reset training data by deleting `TrainingImage/` and `TrainingImageLabel/`

## Common Pitfalls
- dlib requires C++ compiler; use bundled .whl for Python 3.10/Windows
- Camera access fails silently; test with `cv2.VideoCapture(0, cv2.CAP_DSHOW).isOpened()`
- Paths are resolved from `app` directory; run from the application root
- Lighting and face visibility strongly affect recognition quality
- Training is synchronous and blocks until complete
- No authentication; web UI is publicly accessible

## Key Files for Reference
- [app.py](app.py): Main application logic and route handlers
- [face_engine.py](face_engine.py): Face capture, training, recognition, and model utilities
- [database.py](database.py): Database schema for Student and Attendance
- [templates/register.html](templates/register.html): Student registration flow and webcam capture
- [templates/attendance.html](templates/attendance.html): Attendance session flow and instructions
- [static/style.css](static/style.css): Modern UI styling and animations

## Development Notes
- Register students first, then use the attendance page after training
- Dashboard shows the number of students currently present in the trained model
- For best results, keep face centered and avoid backlighting during recognition
- If recognition fails, retrain the model after capturing fresh images
- The app now logs recognized student IDs and confidence values for debugging

## AI Agent Guidance
- Use this file to understand the project scope, not as runtime documentation.
- The codebase is a Flask app with face recognition in `face_engine.py`, DB models in `database.py`, and templates in `templates/`.
- When asked to modify behavior, update `app.py`, `face_engine.py`, or templates based on the requested feature.
- Preserve compatibility with Windows camera access and prebuilt `dlib` wheel if referenced.
- Keep changes concise, test the updated file paths, and avoid adding unrelated frameworks.

## Recent Changes
- Added browser-based webcam preview and face capture for registration
- Updated attendance flow to use live OpenCV camera session with better direct feedback
- Lowered recognition threshold to 80 for more reliable matches
- Changed dashboard to count actual trained students from `labels.pkl`
- Added automatic cleanup and reset capability for fresh model retraining</content>
<parameter name="filePath">c:\Users\Haider Ali\Documents\Attendance-Management-system-using-face-recognition\app\AGENTS.md