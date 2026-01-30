import cv2

model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_path, model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame,(300,300)),
        1.0,
        (300,300),
        (104.0,177.0,123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence >0.6:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            x1,y1,x2,y2  = box.astype("int")

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (x1,y1 -10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        cv2.imshow("Face Detection - Press Q to Exit",frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()