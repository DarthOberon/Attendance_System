# OCR Script 
import cv2
import pytesseract
import os
import re


image_path = "E:\AryanWork_10\IDT_ATTENDAC\input_images\sample.jpeg"

if not os.path.exists(image_path):
    print("Image not found. Place sample.jpeg inside input_images folder.")
    exit()

image = cv2.imread(image_path)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

osd = pytesseract.image_to_osd(gray)
angle = int(re.search('Rotate: (\d+)',osd).group(1))

if angle != 0:
    if angle == 90:
        gray = cv2.rotate(gray,cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        gray = cv2.rotate(gray, cv2.ROTATE_180)
    elif angle == 270:
        gray = cv2.rotate(gray,cv2.ROTATE_90_COUNTERCLOCKWISE)


# print(osd)

# gray = cv2.medianBlur(gray,3)

# thresh  = cv2.adaptiveThreshold(
#     gray, 255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11,2
# )

# _, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

custom_config = r'--oem 3 --psm 6'

text = pytesseract.image_to_string(gray,config= custom_config) 

print("\nExtracted Text:")
print("--------------------------")
print(text)