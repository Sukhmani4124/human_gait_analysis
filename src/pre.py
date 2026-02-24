import cv2

# -----------------------------------
# Dashboard (ONLY VISUALIZATION)
# -----------------------------------
def build_dashboard(original, gray, mask):

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    h, w = original.shape[:2]
    gray_bgr = cv2.resize(gray_bgr, (w, h))
    mask_bgr = cv2.resize(mask_bgr, (w, h))

    top = cv2.hconcat([original, gray_bgr])
    bottom = cv2.hconcat([mask_bgr, mask_bgr])
    dashboard = cv2.vconcat([top, bottom])

    return dashboard


# -----------------------------------
# INPUT VIDEO
# -----------------------------------
video_path = "/Users/sukhmanikaur/Documents/codes/human_gait_analysis/human_gait_analysis/data/CV_dataset/demo.mp4"
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
if not ret:
    print("Video not found.")
    exit()

# -----------------------------------
# PREPROCESSING START
# -----------------------------------

# STEP 1: GRAYSCALE
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# STEP 2: FILTERING
prev_gray = cv2.GaussianBlur(prev_gray, (5,5), 0)

# STEP 3: HISTOGRAM ENHANCEMENT
prev_gray = cv2.equalizeHist(prev_gray)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # GRAYSCALE
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # FILTERING
    # -----------------------------
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # -----------------------------
    # ENHANCEMENT
    # -----------------------------
    gray = cv2.equalizeHist(gray)

    # -----------------------------
    # BACKGROUND SUBTRACTION
    # -----------------------------
    diff = cv2.absdiff(prev_gray, gray)

    # -----------------------------
    # THRESHOLDING
    # -----------------------------
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # MORPHOLOGICAL NOISE REMOVAL
    # -----------------------------
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    dashboard = build_dashboard(frame, gray, thresh)
    cv2.imshow("PREPROCESSING ONLY - Gait Project", dashboard)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()