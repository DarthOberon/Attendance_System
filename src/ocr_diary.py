import cv2
import re
import pytesseract

def extract_roll_numbers(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

    text = pytesseract.image_to_string(gray, config=custom_config)

    roll_numbers = re.findall(r'\b\d{5,}\b', text)

    roll_numbers = list(set(roll_numbers))

    return roll_numbers

if __name__ == "__main__":
    image_path = "E:\AryanWork_10\IDT_ATTENDAC\input_images\sample2.jpeg"
    numbers = extract_roll_numbers(image_path)

    print("Extracted Roll Numbers:")
    print(numbers)
