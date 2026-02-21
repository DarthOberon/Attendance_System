import cv2
import numpy as np
import easyocr
import re

# -----------------------------
# CONFIG
# -----------------------------
CONFIDENCE_THRESHOLD = 0.40
MIN_CONTOUR_AREA = 150
# -----------------------------

reader = easyocr.Reader(['en'], gpu=False)

def remove_horizontal_lines(binary_img):
    """
    Removes notebook horizontal lines using morphology.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cleaned = cv2.subtract(binary_img, detected_lines)
    return cleaned


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    # Remove notebook lines
    cleaned = remove_horizontal_lines(thresh)

    # Slight dilation to strengthen digits
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    return cleaned


def extract_diary_numbers(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return []

    processed = preprocess_image(image)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_numbers = []

    # Sort contours top to bottom
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)

        if area < MIN_CONTOUR_AREA:
            continue

        # Crop digit region from original image
        roi = image[y:y+h, x:x+w]

        # Resize for better OCR
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        results = reader.readtext(roi, allowlist='0123456789')

        for (bbox, text, confidence) in results:

            print(f"Detected: {text} | Confidence: {confidence:.2f}")

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            clean_text = re.sub(r"\D", "", text)

            if clean_text.isdigit() and 1 <= len(clean_text) <= 3:
                normalized = clean_text.zfill(3)
                extracted_numbers.append(normalized)

    extracted_numbers = list(set(extracted_numbers))

    return extracted_numbers


if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sample5.jpeg"

    numbers = extract_diary_numbers(image_path)

    print("\nFinal Normalized Numbers:")
    print(numbers)