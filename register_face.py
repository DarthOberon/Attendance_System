import cv2
import os

student_name = "Student_1"

save_dir = os.path.join("dataset",student_name)
os.makedirs(save_dir, exist_ok=True)

model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

cap = cv2.VideoCapture(0)
count = 0

print("Press S to save face | Press Q or ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h ,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame,(600,600)),
        1.0,
        (300,300),
        (104.0,177.0,123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.6:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            x1,y1,x2,y2 = box.astype("int")

            face = frame[y1:y2,x1:x2]

            if face.size == 0:
                continue
            
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.imshow("Face Registration",frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                if (x2 - x1) >= 120 and (y2 - y1) >= 120:
                    face_resized = cv2.resize(face,(200,200))
                    cv2.imwrite(f"{save_dir}/{count}.jpg",face_resized)
                    print(f"Saved image {count}")
                    count +=1
                else:
                    print("Face is too small- move closer")

            if key == 27 or key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Registration complete")
                exit()

cv2.imshow("Face Registration",frame)
key = cv2.waitKey(1)