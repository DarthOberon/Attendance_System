from paddleocr import PaddleOCR
import cv2
import re

def extract_numbers_paddle(image_path):
    ocr = PaddleOCR(use_textline_orientation=True,lang='en')

    results = ocr.ocr(image_path)

    extracted_numbers = []

    for line in results:
        for word_info in line:
            text = word_info[1][0]
            confidence = word_info[1][1]

            print(f"Detected: {text} | Confidence: {confidence:.2f}")

            # Remove non-digits
            clean_text = re.sub(r'\D', '', text)

            if confidence < 0.5:
                continue

            # Accept only 1-3 digit numbers
            if clean_text.isdigit() and 1 <= len(clean_text) <= 3:
                normalized = clean_text.zfill(3)
                extracted_numbers.append(normalized)

    # Remove duplicates
    extracted_numbers = list(set(extracted_numbers))

    return extracted_numbers


if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sample5.jpeg"

    numbers = extract_numbers_paddle(image_path)

    print("\nFinal Normalized Numbers:")
    print(numbers)