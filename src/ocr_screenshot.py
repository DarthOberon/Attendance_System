import cv2
import pytesseract
import re

def process_digital_screenshot(image_path):
    print("[INFO] Processing Digital Screenshot with Tesseract...")
    
    # 1. Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return []

    # 2. Convert BGR to RGB (Tesseract's preferred color format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Extract text using Tesseract
    # --psm 6 is highly optimized for reading lists/blocks of text
    custom_config = r'--psm 6'
    raw_text = pytesseract.image_to_string(img_rgb, config=custom_config)
    
    print("\n--- RAW TESSERACT OUTPUT ---")
    print(raw_text.strip())
    print("----------------------------\n")
    
    # 4. Clean and Extract Numbers using Regex
    potential_numbers = re.findall(r'\d+', raw_text)
    extracted_numbers = []
    
    for num in potential_numbers:
        length = len(num)
        
        # --- THE COLLEGE BUSINESS RULES ---
        # Only accept 1, 2, 3 digit shorthand, or full 13-digit enrollment
        if 1 <= length <= 3 or length == 13:
            extracted_numbers.append(num)
            print(f"  -> ✅ Validated: {num}")
        else:
            print(f"  -> ❌ Ignored Noise: {num}")

    # Remove duplicates and sort them logically (converting to int for proper numerical sorting)
    extracted_numbers = sorted(list(set(extracted_numbers)), key=int)
    
    return [str(num) for num in extracted_numbers]

if __name__ == "__main__":
    # Point this to the screenshot you just took
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\screenshot_test.png" 
    
    present_students = process_digital_screenshot(image_path)
    
    print("\n✅ Final Present List to send to Database:")
    print(present_students)