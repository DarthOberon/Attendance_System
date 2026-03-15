import cv2
import numpy as np
import pytesseract

def detect_grid(image_path):
    print("[INFO] Loading image and hunting for grid lines....")

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return []  # Return empty list instead of None

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel_length = np.array(img).shape[1] // 40

    vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    temp_1         = cv2.erode(thresh, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(temp_1, vertical_kernel, iterations=2)

    temp_2           = cv2.erode(thresh, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(temp_2, horizontal_kernel, iterations=2)

    grid = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    _, grid = cv2.threshold(grid, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 10 and w < img.shape[1] * 0.9:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[1])

    # --- Y-COORDINATE BASED ROW GROUPING ---
    Y_THRESHOLD = 10

    raw_rows = []
    for box in boxes:
        x, y, w, h = box
        placed = False
        for raw_row in raw_rows:
            if abs(raw_row["y"] - y) < Y_THRESHOLD:
                raw_row["boxes"].append(box)
                placed = True
                break
        if not placed:
            raw_rows.append({"y": y, "boxes": [box]})

    rows = []
    for raw_row in raw_rows:
        row = sorted(raw_row["boxes"], key=lambda b: b[0])
        if len(row) >= 3:
            rows.append(row[:3])

    print(f"[INFO] Successfully structured {len(rows) - 1} data rows!")

    # --- COLLECT PRESENT ROLL NUMBERS ---
    # Only present students go into this list.
    # Flask marks everyone NOT in this list as ABSENT.
    present_roll_numbers = []

    for index, row in enumerate(rows[1:], start=1):

        enroll_box = row[0]
        name_box   = row[1]
        status_box = row[2]

        # --- 1. OCR THE ENROLLMENT NUMBER ---
        ex, ey, ew, eh = enroll_box
        roll_no = "UNKNOWN"

        if ew > 10 and eh > 10:
            enroll_crop = gray[ey+2 : ey+eh-6, ex+2 : ex+ew-2]

            _, temp_thresh = cv2.threshold(enroll_crop, 150, 255, cv2.THRESH_BINARY_INV)
            digit_contours, _ = cv2.findContours(temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if digit_contours:
                all_x  = [cv2.boundingRect(c)[0] for c in digit_contours]
                all_y  = [cv2.boundingRect(c)[1] for c in digit_contours]
                all_x2 = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in digit_contours]
                all_y2 = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in digit_contours]

                dx,  dy  = min(all_x),  min(all_y)
                dx2, dy2 = max(all_x2), max(all_y2)

                digit_only = enroll_crop[dy:dy2, dx:dx2]
                digit_only = cv2.copyMakeBorder(digit_only, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            else:
                digit_only = cv2.copyMakeBorder(enroll_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

            digit_only = cv2.resize(digit_only, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            digit_only = np.uint8(digit_only)

            _, enroll_thresh = cv2.threshold(digit_only, 150, 255, cv2.THRESH_BINARY)

            if enroll_thresh.size > 0:
                roll_no = pytesseract.image_to_string(
                    enroll_thresh,
                    config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
                ).strip()

                if not roll_no:
                    roll_no = pytesseract.image_to_string(
                        enroll_thresh,
                        config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
                    ).strip()

                if not roll_no:
                    roll_no = pytesseract.image_to_string(
                        enroll_thresh,
                        config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'
                    ).strip()

        if roll_no == "":
            roll_no = "UNKNOWN"

        # --- 2. OMR THE STATUS BOX ---
        sx, sy, sw, sh = status_box
        margin = 4

        if sh > (margin * 2) and sw > (margin * 2):
            status_crop = thresh[sy+margin : sy+sh-margin, sx+margin : sx+sw-margin]
            ink_pixels  = cv2.countNonZero(status_crop)
            is_present  = ink_pixels > 15

            print(f"Row {index} | Roll [{roll_no}]: {'PRESENT' if is_present else 'ABSENT'} (Ink: {ink_pixels})")

            # Only add to list if present AND roll number was successfully read
            if is_present and roll_no != "UNKNOWN":
                present_roll_numbers.append(roll_no)
        else:
            print(f"Row {index}: Status box too small to read")

    print(f"\n[INFO] Sheet mode complete. Present: {present_roll_numbers}")

    # Return list of present roll numbers — same format as screenshot and diary mode
    return present_roll_numbers


if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sheet_sample6.png"
    present = detect_grid(image_path)
    print(f"\nFinal Present List: {present}")