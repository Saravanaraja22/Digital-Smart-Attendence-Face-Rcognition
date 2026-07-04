import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"
MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")

os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)

def detect_first_face(bgr_image):
    try:
        import mediapipe as mp
        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is not None:
            detector = mp_solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )
            results = detector.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            if results.detections:
                h, w = bgr_image.shape[:2]
                bbox = results.detections[0].location_data.relative_bounding_box
                x = int(max(0, bbox.xmin * w))
                y = int(max(0, bbox.ymin * h))
                width = int(min(w - x, bbox.width * w))
                height = int(min(h - y, bbox.height * h))
                return x, y, width, height
    except Exception:
        pass

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face[2] * face[3])

# ---- Utility: extract face crop -> small grayscale vector (embedding) ----
def crop_face_and_embed(bgr_image, face_box):
    h, w = bgr_image.shape[:2]
    x, y, width, height = face_box
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(w, x + width))
    y2 = int(min(h, y + height))
    if x2 <= x1 or y2 <= y1:
        return None
    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32,32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32) / 255.0
    return emb

def extract_embedding_for_image(stream_or_bytes):
    # accepts a file-like stream (werkzeug FileStorage.stream)
    # read image from stream into numpy BGR
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    face_box = detect_first_face(img)
    if face_box is None:
        return None
    emb = crop_face_and_embed(img, face_box)
    return emb

# ---- Load model helpers ----
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_with_model(clf, emb):
    # returns label and confidence (max probability)
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf

# ---- Training function used in background ----
def train_model_background(dataset_dir, progress_callback=None):
    """
    dataset_dir/
        student_id/
            img1.jpg
            img2.jpg
    progress_callback(progress_percent, message) -> optional
    """
    X = []
    y = []
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_students = max(1, len(student_dirs))
    processed = 0

    for sid in student_dirs:
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            face_box = detect_first_face(img)
            if face_box is None:
                continue
            emb = crop_face_and_embed(img, face_box)
            if emb is None:
                continue
            X.append(emb)
            y.append(int(sid))
        processed += 1
        if progress_callback:
            pct = int((processed/total_students)*80)  # training progress up to 80% during feature extraction
            progress_callback(pct, f"Processed {processed}/{total_students} students")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    # convert
    X = np.stack(X)
    y = np.array(y)

    # fit RandomForest
    if progress_callback:
        progress_callback(85, "Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    if progress_callback:
        progress_callback(100, "Training complete")
