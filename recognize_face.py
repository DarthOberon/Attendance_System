import cv2
import os
import numpy as np
import csv
from datetime import datetime
import time


DATASET_DIR = "dataset"
MODEL_PATH = "models"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 60
MIN_FACE_SIZE = 120
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720


face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
last_print_time = 0



faces = []
labels = []
label_map = {}
current_label = 0

print("Loading dataset...")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR,person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name
    print(f"Training for: {person_name}")

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path,image_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(img)
            labels.append(current_label)
    
    current_label += 1

if len(faces) == 0:
    print("ERROR: No training data found!")
    exit()

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

print("Face recognition model trained")

cap = cv2.VideoCapture("https://192.168.0.108:8080/video")

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) ---- webcam setting

attendance_file = "attendance.csv"
marked_present = set()

if not os.path.exists(attendance_file):
    with open(attendance_file, "w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name","Date","Time","Status"])

print("System Ready. Press Q or ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    predictions = []

    # -------- PHASE 1: PREDICT ALL FACES --------
    for (x, y, w, h) in detected_faces:

        # Minimum face size check
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            predictions.append({
                "name": "Unknown",
                "conf": None,
                "box": (x, y, w, h)
            })
            continue

        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (200, 200))

        label, confidence = recognizer.predict(face_gray)
        predicted_name = label_map.get(label, "Unknown")

        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_name = "Unknown"

        predictions.append({
            "name": predicted_name,
            "conf": confidence,
            "box": (x, y, w, h)
        })

    # -------- PHASE 2: CONFLICT DETECTION --------
    name_counts = {}
    for pred in predictions:
        if pred["name"] != "Unknown":
            name_counts[pred["name"]] = name_counts.get(pred["name"], 0) + 1

    # -------- PHASE 3: FINAL DECISION --------
    current_time = time.time()

    for pred in predictions:
        name = pred["name"]
        confidence = pred["conf"]
        (x, y, w, h) = pred["box"]

        if name == "Unknown":
            display_text = "Unknown"
            box_color = (0, 0, 255)

        elif name_counts.get(name, 0) > 1:
            display_text = "AMBIGUOUS"
            box_color = (0, 0, 255)

            if current_time - last_print_time > 1:
                print(f"[WARN] Conflict detected for {name} — attendance blocked")

        else:
            display_text = f"{name} ({confidence:.1f})"
            box_color = (0, 255, 0)

            if name not in marked_present:
                now = datetime.now()
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        "Present"
                    ])
                marked_present.add(name)

            if current_time - last_print_time > 1:
                print(f"[INFO] Recognized: {name}, Confidence: {confidence:.2f}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, display_text, (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        if current_time - last_print_time > 1:
            last_print_time = current_time

    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Smart Attendance System", display_frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
