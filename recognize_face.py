import cv2
import os
import numpy as np
import csv
from datetime import datetime

DATASET_DIR = "dataset"
MODEL_PATH = "models"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR,person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path,image_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_label)
    
    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

print("Face recognition model trained")

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

print("Press Q or ESC to exit.")

attendance_file = "attendance.csv"
marked_present = set()

if not os.path.exists(attendance_file):
    with open(attendance_file, "w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name","Date","Time","Status"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in detected_faces:
        face_roi = frame[y:y+h, x:x+w]

    # Safety check
        if face_roi.size == 0:
            continue

    # FORCE grayscale conversion HERE
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (200, 200))

    # EXTRA safety: ensure single channel
        if len(face_gray.shape) != 2:
            continue

        label, confidence = recognizer.predict(face_gray)

        name = label_map.get(label, "Unknown")

        if confidence < 65:
            text = f"{name} ({confidence:.1f})"

            if name not in marked_present:
                now = datetime.now()
                with open(attendance_file,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H-%M-%S"),
                        "Present"
                    ])
                marked_present.add(name)

        else:
            text = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    cv2.imshow("Face Recognition - Press Q or ESC",frame)

    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
