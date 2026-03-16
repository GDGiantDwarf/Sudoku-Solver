import cv2
import numpy as np
import os

TEMPLATE_DIR = "templates"
CELL_SIZE = 40

# Pixels trimmed from each side of a cell before digit recognition.
# Removes grid line contamination (including thick 3x3 box boundaries at corners/edges)
# which would otherwise produce contours larger than the digit itself.
CELL_MARGIN = 6

# A valid digit glyph must be at least this fraction of the (margined) cell height.
# Filters out residual noise and any stray tiny marks without affecting real digits.
MIN_DIGIT_HEIGHT_RATIO = 0.40


def preprocess_digit(img):

    img = binarize(img)

    # invert so digits become white
    img = cv2.bitwise_not(img)

    contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros((CELL_SIZE,CELL_SIZE),dtype=np.uint8)

    cell_h = img.shape[0]

    # Keep only contours whose bounding-box height is at least MIN_DIGIT_HEIGHT_RATIO
    # of the cell height. This rejects:
    #   - residual grid-line slivers that survive the margin trim
    #   - any noise or stray marks that are clearly too short to be a digit
    valid = [c for c in contours
             if cv2.boundingRect(c)[3] >= cell_h * MIN_DIGIT_HEIGHT_RATIO]

    if len(valid) == 0:
        return np.zeros((CELL_SIZE,CELL_SIZE),dtype=np.uint8)

    c = max(valid, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)

    digit = img[y:y+h, x:x+w]

    digit = cv2.resize(digit,(CELL_SIZE,CELL_SIZE))

    return digit


def binarize(img):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # strong blur removes UI gradients
    img = cv2.GaussianBlur(img, (5,5), 0)

    # strict threshold -> only 0 or 255
    _, th = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    return th

def load_templates():
    templates = {}

    for i in range(1,10):

        path = os.path.join(TEMPLATE_DIR, f"{i}.png")

        img = cv2.imread(path)

        if img is None:
            raise Exception(f"Template {path} missing")

        img = binarize(img)

        img = preprocess_digit(img)

        templates[i] = img

    return templates


TEMPLATES = load_templates()


def extract_board_from_screenshot(path):

    img = cv2.imread(path)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)

    edges = cv2.Canny(blur,50,150)

    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours,key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)

    if len(approx) != 4:
        raise Exception("Could not detect grid")

    pts = order_points(approx.reshape(4,2))
    warped = perspective_transform(gray, pts)
    warped = binarize(warped)

    warped_color = cv2.cvtColor(warped,cv2.COLOR_GRAY2BGR)

    size = warped.shape[0]
    cell = size // 9

    # Cell (0,0) centre in original screenshot coordinates.
    # pts is ordered [tl, tr, br, bl]; we step half a cell along each axis.
    tl, tr, _, bl = pts
    right_vec = (tr - tl) / 9.0
    down_vec  = (bl - tl) / 9.0
    cell00_center = tl + 0.5 * right_vec + 0.5 * down_vec
    cell00_screen = (int(round(cell00_center[0])), int(round(cell00_center[1])))

    board = [[0]*9 for _ in range(9)]

    for r in range(9):
        for c in range(9):

            y1 = r*cell + CELL_MARGIN
            y2 = (r+1)*cell - CELL_MARGIN
            x1 = c*cell + CELL_MARGIN
            x2 = (c+1)*cell - CELL_MARGIN

            cell_img = warped[y1:y2,x1:x2]

            digit = recognize_digit(cell_img)

            board[r][c] = digit

            if digit != 0:
                cv2.putText(
                    warped_color,
                    str(digit),
                    (x1+cell//3,y1+2*cell//3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )

    # draw grid lines
    for i in range(10):
        p = i*cell
        cv2.line(warped_color,(0,p),(size,p),(0,255,0),1)
        cv2.line(warped_color,(p,0),(p,size),(0,255,0),1)

    cv2.imwrite("grid_debug.png",warped_color)

    print("\nDetected Sudoku board:\n")
    for row in board:
        print(row)
    print(f"\nCell (0,0) screen centre: {cell00_screen}")

    return board, cell00_screen


def recognize_digit(cell):

    digit_img = preprocess_digit(cell)

    best_digit = 0
    best_score = 0

    for digit,template in TEMPLATES.items():

        res = cv2.matchTemplate(digit_img,template,cv2.TM_CCOEFF_NORMED)
        score = res.max()

        if score > best_score:
            best_score = score
            best_digit = digit

    if best_score > 0.55:
        return best_digit

    return 0



def perspective_transform(img,pts):

    tl,tr,br,bl = pts

    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxW = int(max(widthA,widthB))

    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxH = int(max(heightA,heightB))

    dst = np.array([
        [0,0],
        [maxW-1,0],
        [maxW-1,maxH-1],
        [0,maxH-1]
    ],dtype="float32")

    M = cv2.getPerspectiveTransform(pts,dst)

    return cv2.warpPerspective(img,M,(maxW,maxH))


def order_points(pts):

    rect = np.zeros((4,2),dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect