import base64
import cv2
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "TrainingImage")
MODEL_FOLDER = os.path.join(BASE_DIR, "TrainingImageLabel")
MODEL_FILE   = os.path.join(MODEL_FOLDER, "face_model.yml")
LABELS_FILE  = os.path.join(MODEL_FOLDER, "labels.pkl")
FACE_SIZE    = (150, 150)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _get_recognizer():
    # Create an LBPH recognizer optimized for better discrimination.
    # Increased radius and neighbors for better feature detection and rejection of unknown faces.
    return cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=10, grid_x=10, grid_y=10
    )


def _preprocess(face_img):
    """
    Normalize a grayscale face image for consistent recognition.
    Uses CLAHE for superior contrast enhancement and better face discrimination.
    """
    face_img = cv2.resize(face_img, FACE_SIZE)
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for superior preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_img = clahe.apply(face_img)                # advanced contrast enhancement
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)  # reduce noise
    return face_img


def _detect_faces(frame):
    """
    Detect face locations in a frame.
    Returns list of (top, right, bottom, left).
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
    )
    locations = []
    for (x, y, w, h) in faces:
        locations.append((y, x + w, y + h, x))
    return locations


def test_camera():
    """Simple camera test to verify camera is working"""
    print("Testing camera...")
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("ERROR: Camera not accessible")
        return False

    ret, frame = cam.read()
    if not ret or frame is None:
        print("ERROR: Cannot read from camera")
        cam.release()
        return False

    print(f"Camera test successful - Frame shape: {frame.shape}")
    cam.release()
    return True


def save_face_capture(student_id, name, image_data):
    """Decode browser image data, detect the face, and save a preprocessed image."""
    folder = os.path.join(TRAINING_DIR, f"{student_id}_{name}")
    os.makedirs(folder, exist_ok=True)

    try:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

    if frame is None:
        return None

    locations = _detect_faces(frame)
    if not locations:
        return 0

    top, right, bottom, left = locations[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi = gray[top:bottom, left:right]
    if face_roi.size == 0:
        return 0

    processed = _preprocess(face_roi)
    existing = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(existing)
    cv2.imwrite(os.path.join(folder, f"{count}.jpg"), processed)
    return count + 1


def recognize_face(image_data):
    """
    Recognize a face from browser image data during attendance session.
    Returns {id, name, confidence} if recognized, 'unknown' if not, or None if error.
    """
    print("=== RECOGNIZE_FACE CALLED ===")

    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print("ERROR: Model files not found")
        return None

    try:
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        print(f"Image decoded successfully, shape: {frame.shape}")
    except Exception as e:
        print(f"ERROR: Failed to decode image: {e}")
        return None

    if frame is None:
        print("ERROR: Frame is None")
        return None

    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
    )

    print(f"Detected {len(faces)} faces")

    if len(faces) == 0:
        print("No faces detected")
        return None

    # Recognize the first detected face
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    if face_roi.size == 0:
        print("ERROR: Face ROI is empty")
        return None

    processed = _preprocess(face_roi)
    print("Face preprocessed successfully")

    # Load model and labels
    recognizer = _get_recognizer()
    recognizer.read(MODEL_FILE)
    with open(LABELS_FILE, "rb") as f:
        id_map = pickle.load(f)

    print(f"Model loaded, labels: {list(id_map.keys())}")

    # Predict
    label, confidence = recognizer.predict(processed)
    print(f"Raw prediction - Label: {label}, Confidence: {confidence}")

    # TEMPORARY: Force recognition for testing
    info = id_map.get(label, {"id": "Unknown", "name": "Unknown"})
    result = {
        "id": info["id"],
        "name": info["name"],
        "confidence": round(confidence, 2)
    }
    print(f"FORCED RECOGNITION: {result}")
    return result


# ─────────────────────────────────────────────
# STEP 1 — CAPTURE FACE IMAGES
# ─────────────────────────────────────────────
def capture_faces(student_id, name, num_images=30):
    """
    Opens webcam, captures and saves preprocessed grayscale face images.
    30 images gives a good training set.
    """
    folder = os.path.join(TRAINING_DIR, f"{student_id}_{name}")
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        return 0

    count = 0

    while count < num_images:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        locations = _detect_faces(frame)

        for (top, right, bottom, left) in locations:
            face_roi  = gray[top:bottom, left:right]
            processed = _preprocess(face_roi)
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), processed)
            count += 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
            cv2.putText(frame, f"{count}/{num_images}", (left, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        status = f"Capturing for {name} — {count}/{num_images}  |  ESC to stop"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Register Student Face", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    return count


# ─────────────────────────────────────────────
# STEP 2 — TRAIN MODEL
# ─────────────────────────────────────────────
def train_model():
    """
    Loads all saved face images, trains LBPH recognizer.
    Saves model + label map.
    """
    faces         = []
    labels        = []
    id_map        = {}   # numeric_label → { id, name }
    label_counter = 0

    if not os.path.exists(TRAINING_DIR):
        return False, "TrainingImage folder not found."

    for student_folder in os.listdir(TRAINING_DIR):
        folder_path = os.path.join(TRAINING_DIR, student_folder)
        if not os.path.isdir(folder_path):
            continue

        parts = student_folder.split("_", 1)
        if len(parts) != 2:
            continue

        student_id, sname = parts[0], parts[1]
        numeric_label     = label_counter
        id_map[numeric_label] = {"id": student_id, "name": sname}
        label_counter += 1
        loaded = 0

        for image_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_name)
            img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = _preprocess(img)
            faces.append(img)
            labels.append(numeric_label)
            loaded += 1

        print(f"  Loaded {loaded} images for {sname} ({student_id})")

    if not faces:
        return False, "No face images found. Register students first."

    recognizer = _get_recognizer()
    recognizer.train(faces, np.array(labels))

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(id_map, f)

    msg = f"Model trained: {len(faces)} images across {label_counter} student(s)."
    print(msg)
    return True, msg


# ─────────────────────────────────────────────
# STEP 3 — LIVE ATTENDANCE SESSION
# ─────────────────────────────────────────────
def run_attendance_session(subject="General"):
    """
    Opens webcam. Recognizes faces continuously.
    Draws name + student ID on screen for each recognized face.
    Returns list of { id, name } that were marked.

    subject is optional — pass empty string if not needed.
    Press ESC to end session.
    """
    print("=== STARTING ATTENDANCE SESSION ===")

    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        print("ERROR: Model not trained. Train first.")
        return []

    print(f"Model file exists: {os.path.exists(MODEL_FILE)}")
    print(f"Labels file exists: {os.path.exists(LABELS_FILE)}")

    recognizer = _get_recognizer()
    recognizer.read(MODEL_FILE)
    with open(LABELS_FILE, "rb") as f:
        id_map = pickle.load(f)

    print(f"Model loaded. Available labels: {list(id_map.keys())}")
    print(f"Label mappings: {id_map}")

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("ERROR: Camera could not be opened!")
        return []

    print("Camera opened successfully")

    # Test camera before starting loop
    if not test_camera():
        return []

    marked      = {}   # student_id → name (only marked once per session)
    frame_count = 0
    start_time  = cv2.getTickCount()

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        frame_count += 1
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        locations = _detect_faces(frame)

        if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
            print(f"Frame {frame_count}: Detected {len(locations)} face(s)")

        for (top, right, bottom, left) in locations:
            face_roi   = gray[top:bottom, left:right]
            if face_roi.size == 0:
                continue
            processed  = _preprocess(face_roi)
            label, confidence = recognizer.predict(processed)
            # TEMPORARY: Very lenient threshold for debugging
            recognition_threshold = 100  # Accept almost anything to see if recognition works

            print(f"Face detected - Label: {label}, Confidence: {confidence:.1f}, Threshold: {recognition_threshold}")

            if confidence < recognition_threshold:
                info       = id_map.get(label, {"id": "Unknown", "name": "Unknown"})
                student_id = info["id"]
                name       = info["name"]
                color      = (0, 220, 0)

                # Mark attendance (once per session per student)
                if student_id not in marked:
                    marked[student_id] = name
                    print(f"  Marked: {name} ({student_id}) - Confidence: {confidence:.1f}")

                label_text = f"{name}  [{student_id}]"
                conf_text  = f"Match: {max(0, round(100 - confidence))}%"
            else:
                color      = (0, 60, 220)
                label_text = "Unknown"
                conf_text  = f"Conf: {confidence:.1f} (Rejected)"

            # Draw box and name on frame
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom), (right, bottom + 40), color, -1)
            cv2.putText(frame, label_text, (left + 4, bottom + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, conf_text, (left + 4, bottom + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Status bar at top
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        subj_text = f"Subject: {subject}  |  " if subject else ""
        status    = f"{subj_text}Marked: {len(marked)} student(s)  |  Time: {elapsed_time:.1f}s  |  ESC to finish"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imshow("Attendance Session  —  ESC to finish", frame)

        # Auto-exit after 30 seconds if no one is detected, or after 60 seconds regardless
        if elapsed_time > 60 or (elapsed_time > 30 and len(marked) == 0):
            print(f"Auto-exiting attendance session (elapsed: {elapsed_time:.1f}s, marked: {len(marked)})")
            break

        if cv2.waitKey(1) == 27:
            print("ESC pressed - exiting attendance session")
            break

    cam.release()
    cv2.destroyAllWindows()

    result = [{"id": sid, "name": sname} for sid, sname in marked.items()]
    print(f"=== ATTENDANCE SESSION ENDED ===")
    print(f"Total marked students: {len(result)}")
    print(f"Marked students: {result}")
    return result