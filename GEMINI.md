# Face Recognition Attendance System - Instructions

This document provides foundational guidance for the Face Recognition Attendance System project. Adhere to these standards and workflows to ensure consistency and system integrity.

## Project Overview
A Flask-based application for automated attendance management using facial recognition and liveness detection.

## Tech Stack
- **Backend:** Flask (Python)
- **Database:** SQLite with Flask-SQLAlchemy
- **Computer Vision:** OpenCV (LBPH Recognizer, Haar Cascades)
- **Security:** dlib (68-landmark predictor for eye-blink liveness detection)
- **Frontend:** Jinja2 Templates, Vanilla CSS, Modern JavaScript (Fetch API)

## Architecture & Modular Breakdown
- `app.py`: Main Flask application, route definitions, and service coordination.
- `database.py`: SQLAlchemy models (`Student`, `Attendance`).
- `face_engine.py`: Face detection, preprocessing (CLAHE), training (LBPH), and recognition logic.
- `liveness.py`: Liveness detection using dlib landmarks to detect eye blinks.
- `templates/`: UI templates extending `base.html`.
- `TrainingImage/`: Storage for captured face images (categorized by `student_id_name`).
- `TrainingImageLabel/`: Storage for the trained model (`face_model.yml`) and label mappings (`labels.pkl`).

## Core Workflows

### 1. Student Registration
- Register student details in the database.
- Capture ~30-50 grayscale, preprocessed face images via the browser/webcam.
- Save images in `TrainingImage/{student_id}_{name}/`.

### 2. Model Training
- Read all images from `TrainingImage/`.
- Preprocess using CLAHE for contrast normalization.
- Train the `LBPHFaceRecognizer`.
- Save the resulting model and a pickle file mapping numeric labels to Student IDs.

### 3. Attendance Marking
- **Liveness Check:** Use `liveness.py` to verify the user is a live human via eye-blink detection.
- **Recognition:** Capture a frame, detect face, preprocess, and predict using the trained model.
- **Database Update:** Log the recognized student's attendance if not already marked for the day.

## Coding Conventions
- **Preprocessing:** Always use the `_preprocess` function in `face_engine.py` (CLAHE + Gaussian Blur) for both training and recognition to maintain consistency.
- **Database:** Use `db.session` for all database interactions. Ensure `app_context` is used when running scripts outside the request cycle.
- **Error Handling:** Return consistent JSON responses (`{status: 'ok'|'error', msg: '...'}`) for all API endpoints.
- **UI/UX:** Maintain the modern, card-based aesthetic defined in `style.css`. Use "Quick Actions" for primary dashboard tasks.

## Security & Reliability
- **Liveness is Mandatory:** Do not bypass the liveness check for attendance sessions unless explicitly instructed for debugging.
- **Data Protection:** Never log or expose raw image data. Images are stored locally and should be treated as sensitive data.
- **Confidence Thresholds:** The system uses LBPH confidence scores. Lower scores mean better matches. Adjust thresholds carefully in `face_engine.py`.

## Local Setup Requirements
- `dlib-19.22.99-cp310-cp310-win_amd64.whl`: Pre-built dlib wheel for Windows.
- `shape_predictor_68_face_landmarks.dat`: Required for liveness detection.
- `haarcascade_frontalface_default.xml`: Required for face detection.
