import cv2
import numpy as np
import pytesseract

def detect_grid(image_path):
    print("[INFO] loading image and hunting for grid lines....")

    img  = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT, value=[255,255,255])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)

    kernel_length = np.array(img).shape[1] // 40

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_length,1))

    temp_1 = cv2.erode(thresh,vertical_kernel,iterations=2)
    vertical_lines = cv2.dilate(temp_1, vertical_kernel, iterations=2)

    temp_2 = cv2.erode(thresh,horizontal_kernel,iterations=2)
    horizontal_lines = cv2.dilate(temp_2, horizontal_kernel, iterations=2)

    grid = cv2.addWeighted(vertical_lines,0.5,horizontal_lines,0.5,0.0)
    _, grid = cv2.threshold(grid,50,255,cv2.THRESH_BINARY)

    contours, heirarchy = cv2.findContours(grid, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    boxes= []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if w > 30 and h > 15 and w < img.shape[1] *0.9:
            boxes.append((x,y,w,h))

    boxes.sort(key=lambda b: b[1])

    rows = []
    for i in range(0, len(boxes),3):
        row = boxes[i:i+3]
        row.sort(key=lambda b: b[0])
        rows.append(row)

    print(f"\n Successfully structured {len(rows)} rows!")
    print("\n--------Attendance Scan Results-----------")


    for index, row in enumerate(rows[1:],start=1):
        
        if len(row) != 3:
            continue

        enroll_box = row[0]
        name_box = row[1]
        status_box = row[2]

        sx,sy,sw,sh = status_box
        
        margin = 4

        if sh>(margin * 2) and sw > (margin * 2):
            status_crop = thresh[sy+margin : sy+sh-margin,sx+margin:sx+sw-margin]

        # _, crop_thresh = cv2.threshold(status_crop,150,255,cv2.THRESH_BINARY_INV)

            ink_pixels = cv2.countNonZero(status_crop)

            is_present = ink_pixels >15

            if is_present:
                print(f"Row {index}: Present (Ink pixels detected: {ink_pixels})")
            else:
                print(f"Row {index}: Absent (Empty box, pixels: {ink_pixels})")
        else:
            print(f"Row {index}: Box too small to read")

    boxes_img = img.copy() 
    for x,y,w,h in boxes:
        cv2.rectangle(boxes_img,(x,y),(x+w,y+h),(0,255,0),2)

    # cv2.imshow("1.Original Image",img)
    # cv2.imshow("2.Vertical lines only",vertical_lines)
    # cv2.imshow("3.Horizontal line only",horizontal_lines)
    # cv2.imshow("4.The Extracted grid",grid)
    cv2.imshow("5.Detected Cells (Target_locked)",boxes_img)

    print("Success! Press any key on the image windows to close them.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sheet_sample4.png"
    detect_grid(image_path)