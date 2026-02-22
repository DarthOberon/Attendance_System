import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("digit_model.h5")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    return thresh, image

def extract_digits(image_path):
    thresh, original = preprocess_image(image_path)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w*h < 150:
            continue

        roi = thresh[y:y+h, x:x+w]
        roi = cv2.resize(roi, (28,28))
        roi = roi / 255.0
        roi = roi.reshape(1,28,28)

        prediction = model.predict(roi, verbose=0)
        digit = np.argmax(prediction)

        confidence = np.max(prediction)

        print(f"Detected digit: {digit} | Confidence: {confidence:.2f}")

        digit_regions.append((x, y, digit))

    # Sort top-to-bottom, then left-to-right
    digit_regions = sorted(digit_regions, key=lambda x: (x[1], x[0]))

    # Combine nearby digits into numbers
    numbers = []
    current_number = ""
    last_y = -100

    for x, y, digit in digit_regions:
        if abs(y - last_y) > 25:
            if current_number != "":
                numbers.append(current_number.zfill(3))
            current_number = str(digit)
        else:
            current_number += str(digit)
        last_y = y

    if current_number != "":
        numbers.append(current_number.zfill(3))

    return list(set(numbers))


if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sample5.jpeg"

    result = extract_digits(image_path)

    print("\nFinal Roll Numbers:")
    print(result)