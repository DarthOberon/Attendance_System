import easyocr
import cv2
import re

def extract_handwritten_numbers(image_path):
    reader = easyocr.Reader(['en'],gpu=False)

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(gray, None, fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

    # _, thresh = cv2.threshold(resized,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    results = reader.readtext(image_path,allowlist='0123456789',adjust_contrast=True)
    extracted_numbers =[]

    for (bbox,text,confidence) in results:
        print(f"Detected: {text} | Confidence: {confidence:.2f}")

        if confidence < 0.5:
            continue

        clean_text = text.replace(" ","")
        
        if clean_text.isdigit():
            extracted_numbers.append(clean_text)

    extracted_numbers = list(set(extracted_numbers))

    return extracted_numbers

if __name__ == "__main__":
    image_path = "E:\AryanWork_10\IDT_ATTENDAC\input_images\sample2.jpeg"
    numbers = extract_handwritten_numbers(image_path)

    print("\nFinal Extracted Number:")
    print(numbers)