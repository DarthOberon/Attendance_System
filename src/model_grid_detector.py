import cv2
import numpy as np
import pytesseract
import os



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cell_threshold(cell_gray):
    """
    Returns a fixed threshold value tuned to this sheet's ink/paper contrast.
    Ink: ~77-130  |  Paper background: ~160-180
    Uses (cell_mean - 28), clamped to [100, 150].
    """
    thresh = int(np.mean(cell_gray) - 28)
    return max(100, min(150, thresh))


def ocr_enrollment_cell(cell_gray):
    """
    Read the enrollment/roll number from a single cell crop.
    Returns a digit string, or 'UNKNOWN' if nothing readable was found.
    """
    h, w = cell_gray.shape
    if h < 5 or w < 5:
        return "UNKNOWN"

    thresh_val = _cell_threshold(cell_gray)

    # Binarise: ink -> white, background -> black
    _, binary = cv2.threshold(cell_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Collect ink contours, ignoring grid-line artefacts
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if y < 3:           continue   # touches top edge -> grid line
        if cw > w * 0.85:   continue   # spans full width  -> grid artefact
        if area < 8:        continue   # speck noise
        valid.append(c)

    if not valid:
        return "UNKNOWN"

    # Tight bounding box around all valid ink
    xs  = [cv2.boundingRect(c)[0]                             for c in valid]
    ys  = [cv2.boundingRect(c)[1]                             for c in valid]
    xs2 = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]   for c in valid]
    ys2 = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]   for c in valid]
    tx  = max(0, min(xs)  - 3)
    ty  = max(0, min(ys)  - 3)
    tx2 = min(w, max(xs2) + 3)
    ty2 = min(h, max(ys2) + 3)

    crop = cell_gray[ty:ty2, tx:tx2]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return "UNKNOWN"

    # Upscale + re-binarise (white bg, black text -- Tesseract default)
    crop_up  = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    crop_pad = cv2.copyMakeBorder(crop_up, 25, 25, 25, 25,
                                  cv2.BORDER_CONSTANT, value=255)
    _, crop_bin = cv2.threshold(crop_pad, thresh_val, 255, cv2.THRESH_BINARY)

    # PSM 7 (single text line) first, then 13 (raw line), then 8 (single word)
    for psm in [7, 13, 8]:
        r = pytesseract.image_to_string(
            crop_bin,
            config=f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if r:
            return r

    return "UNKNOWN"


def is_status_present(cell_gray):
    """
    Decide whether a status cell contains a 'P' (present) mark.
    Returns (bool, score).

    Strategy: threshold locally, remove cell-edge pixels (grid lines),
    then look for a compact blob that is NOT a flat horizontal line.
    """
    h, w = cell_gray.shape
    if h < 5 or w < 5:
        return False, 0

    thresh_val = _cell_threshold(cell_gray)
    _, binary = cv2.threshold(cell_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Erase cell borders to eliminate grid-line bleed-through
    border_v = max(4, h // 8)   # top/bottom
    border_h = 5                # left/right
    binary[:border_v, :]  = 0
    binary[-border_v:, :] = 0
    binary[:, :border_h]  = 0
    binary[:, -border_h:] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        # Reject horizontal-line artefacts
        if cw > w * 0.7:    continue   # too wide
        if ch < 8:           continue   # flat  -> grid-line remnant  *** KEY FIX ***
        if ch > h * 0.9:     continue   # too tall -> vertical artefact
        if cw > ch * 3.0:    continue   # aspect: much wider than tall -> line artefact

        hull      = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0

        if solidity > 0.25:
            best_score = max(best_score, area * solidity)

    return best_score >= 80, best_score


# ─────────────────────────────────────────────────────────────────────────────
# LINE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_line_positions(projection, img_span, min_strength_ratio):
    """Return the centre Y (or X) position of each detected line."""
    threshold = img_span * min_strength_ratio
    positions = []
    in_line = False
    start = 0
    for i, v in enumerate(projection):
        if v > threshold and not in_line:
            in_line = True
            start = i
        elif v <= threshold and in_line:
            in_line = False
            positions.append((start + i) // 2)
    return positions


def detect_grid_lines(thresh, img_h, img_w):
    """Returns (h_line_ys, v_line_xs) - sorted lists of grid-line positions."""
    # Horizontal lines
    hk_len  = img_w // 6
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk_len, 1))
    horiz   = cv2.dilate(cv2.erode(thresh, hkernel, iterations=2),
                         hkernel, iterations=2)
    h_line_ys = _find_line_positions(np.sum(horiz, axis=1) / 255, img_w, 0.30)

    # Vertical lines
    vk_len  = img_h // 6
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_len))
    vert    = cv2.dilate(cv2.erode(thresh, vkernel, iterations=2),
                         vkernel, iterations=2)
    v_line_xs = _find_line_positions(np.sum(vert, axis=0) / 255, img_h, 0.30)

    grid = cv2.addWeighted(horiz, 0.5, vert, 0.5, 0.0)
    _, grid = cv2.threshold(grid, 50, 255, cv2.THRESH_BINARY)


    return h_line_ys, v_line_xs



# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_grid(image_path):
    print("[INFO] Loading image...")

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Image not found:", image_path)
        return []

    # Thin white border so lines at the very image edge are detected correctly
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                             cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img_h, img_w = img.shape[:2]
    
    

    # ── PRE-PROCESSING ───────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 10
    )

    # ── GRID LINE DETECTION ──────────────────────────────────────────────────
    h_line_ys, v_line_xs = detect_grid_lines(thresh, img_h, img_w)

    if len(v_line_xs) < 4:
        print("[ERROR] Not enough vertical lines found (need >= 4). Aborting.")
        return []

    # ── COLUMN BOUNDS ────────────────────────────────────────────────────────
    # Typical layout produces 6 V-lines:
    #   [0]=left-outer  [1]=inner-left  [2]=enroll|name  [3]=name|status
    #   [4]=status-right  [5]=right-outer
    if len(v_line_xs) >= 6:
        enroll_x1, enroll_x2 = v_line_xs[1], v_line_xs[2]
        status_x1, status_x2 = v_line_xs[3], v_line_xs[4]
    elif len(v_line_xs) == 5:
        enroll_x1, enroll_x2 = v_line_xs[0], v_line_xs[1]
        status_x1, status_x2 = v_line_xs[2], v_line_xs[3]
    else:   # 4 lines
        enroll_x1, enroll_x2 = v_line_xs[0], v_line_xs[1]
        status_x1, status_x2 = v_line_xs[2], v_line_xs[3]


    # ── ROW BOUNDS ───────────────────────────────────────────────────────────
    if len(h_line_ys) < 3:
        print("[ERROR] Not enough horizontal lines found. Aborting.")
        return []

    all_row_pairs = list(zip(h_line_ys, h_line_ys[1:]))
    # Skip row 0 (the "enrollment number / name / status" header row).
    # Skip slivers (< 20 px — these are double-detected lines).
    # Skip the last row if it sits at the very bottom of the image (table border shadow).
    data_rows = [
        (y1, y2) for (y1, y2) in all_row_pairs[1:]
        if y2 - y1 > 20 and y2 < img_h - 15
    ]
    print(f"[INFO] Found {len(data_rows)} data rows (after skipping header, slivers, bottom edge)")

    if not data_rows:
        print("[ERROR] No data rows found.")
        return []

    # data_rows[0] is the row containing the printed header text
    # ("enrollment number", "name", "status") — skip it, start from index 1.
    student_rows = data_rows[1:]
    print(f"[INFO] Processing {len(student_rows)} student rows")

    present_roll_numbers = []

    for i, (y1, y2) in enumerate(student_rows):
        row_num = i + 1
        pad_e = 5   # enroll cell inner padding
        pad_s = 6   # status cell (slightly bigger to trim grid lines)

        enroll_cell = gray[y1+pad_e : y2-pad_e, enroll_x1+pad_e : enroll_x2-pad_e]
        status_cell = gray[y1+pad_s : y2-pad_s, status_x1+pad_s : status_x2-pad_s]

        # ── Enrollment OCR ──
        roll_no = ocr_enrollment_cell(enroll_cell)

        # Save debug image for each enrollment cell
    

        # ── Status OMR ──
        present, score = is_status_present(status_cell)
        status_str = "PRESENT" if present else "ABSENT "

        print(f"Row {row_num:2d} | Roll [{roll_no:>6}] | {status_str}  (ink_score={score:.0f})")

        if present and roll_no not in ("UNKNOWN", ""):
            present_roll_numbers.append(roll_no)

    print(f"\n[INFO] Sheet scan complete.")
    print(f"[INFO] Present roll numbers: {present_roll_numbers}")
    return present_roll_numbers


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_path = r"E:\AryanWork_10\IDT_ATTENDAC\input_images\sample_2.jpeg"
    present = detect_grid(image_path)
    print(f"\nFinal Present List: {present}")