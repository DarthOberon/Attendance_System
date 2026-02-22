import easyocr
import cv2
import numpy as np
import re

CONF_THRESHOLD = 0.35
LINE_Y_THRESHOLD = 25   # vertical grouping sensitivity
ROLL_MODE = True        # True = roll numbers (1-3 digits)

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Slight resize only
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return gray


def group_by_lines(detections):
    """
    Groups detected text boxes by Y coordinate.
    """
    lines = []

    for det in detections:
        (bbox, text, conf) = det
        y_center = int((bbox[0][1] + bbox[2][1]) / 2)

        placed = False
        for line in lines:
            if abs(line["y"] - y_center) < LINE_Y_THRESHOLD:
                line["items"].append(det)
                placed = True
                break

        if not placed:
            lines.append({"y": y_center, "items": [det]})

    return lines


def normalize_number(num):
    if ROLL_MODE:
        # Accept 1–3 digit numbers only
        if 1 <= len(num) <= 3:
            return num.zfill(3)
    return None


def extract_numbers(image_path):

    img = preprocess_image(image_path)
    if img is None:
        return []

    reader = easyocr.Reader(['en'], gpu=False)

    results = reader.readtext(
        img,
        allowlist='0123456789',
        paragraph=False,
        detail=1,
        mag_ratio=2
    )

    # Filter by confidence
    filtered = []
    for bbox, text, conf in results:
        text = text.replace('l','1').replace('I','1').replace('|','1').replace('L','1')
        print(f"Detected raw: {text} | Conf: {conf:.2f}")
        if conf < CONF_THRESHOLD:
            continue

        clean = re.sub(r'\D', '', text)
        if clean:
            filtered.append((bbox, clean, conf))

    # Group by lines
    lines = group_by_lines(filtered)

    final_numbers = []

    for line in lines:
        # Sort left to right
        sorted_line = sorted(line["items"], key=lambda x: x[0][0][0])

        for bbox, text, conf in sorted_line:
            normalized = normalize_number(text)
            if normalized:
                final_numbers.append(normalized)

    # Remove duplicates
    final_numbers = list(set(final_numbers))

    return sorted(final_numbers)


if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sample13.jpeg"

    numbers = extract_numbers(image_path)

    print("\nFinal Clean Roll Numbers:")
    print(numbers)