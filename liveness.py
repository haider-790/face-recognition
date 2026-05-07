import cv2
import dlib
import numpy as np
import time
import os
from scipy.spatial import distance

PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
LEFT_EYE_IDX      = list(range(42, 48))
RIGHT_EYE_IDX     = list(range(36, 42))
EAR_THRESHOLD     = 0.25
MIN_CLOSED_FRAMES = 2


def _eye_aspect_ratio(eye_pts):
    A = distance.euclidean(eye_pts[1], eye_pts[5])
    B = distance.euclidean(eye_pts[2], eye_pts[4])
    C = distance.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)


def _predictor_available():
    return os.path.exists(PREDICTOR_PATH)


def check_liveness(timeout_seconds=10):

    if not _predictor_available():
        print("WARNING: shape_predictor_68_face_landmarks.dat not found — skipping liveness.")
        return True

    detector      = dlib.get_frontal_face_detector()
    predictor     = dlib.shape_predictor(PREDICTOR_PATH)
    cam           = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    blink_count   = 0
    closed_frames = 0
    start_time    = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            cam.release()
            cv2.destroyAllWindows()
            return False

        ret, frame = cam.read()
        if not ret or frame is None or frame.size == 0:
            continue

        # Convert to RGB contiguous array — fixes dlib "Unsupported image type" error
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = np.ascontiguousarray(
            gray,
            dtype=np.uint8
        )

        faces = detector(gray)


        for face in faces:
            shape = predictor(gray, face)
            pts   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            left_ear  = _eye_aspect_ratio(pts[LEFT_EYE_IDX])
            right_ear = _eye_aspect_ratio(pts[RIGHT_EYE_IDX])
            ear       = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                if closed_frames >= MIN_CLOSED_FRAMES:
                    blink_count += 1
                closed_frames = 0

        color     = (0, 200, 0) if blink_count > 0 else (0, 140, 255)
        msg       = "Blink detected — LIVE!" if blink_count > 0 else "Please blink once"
        time_left = int(timeout_seconds - elapsed)
        cv2.putText(frame, msg,                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Time left: {time_left}s",(10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.imshow("Liveness Check — Blink Once", frame)

        if blink_count >= 1:
            cam.release()
            cv2.destroyAllWindows()
            return True

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    return False