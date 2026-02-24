import cv2
import numpy as np

# kernels created once (faster)
_kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

def resize_frame(frame, max_width=960):
    h,w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame,(max_width,int(h*scale)))

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def create_background_subtractor():
    # ⭐ strong stable silhouette settings
    return cv2.createBackgroundSubtractorMOG2(
        history=120,
        varThreshold=40,
        detectShadows=False
    )

def preprocess_frame(frame, bg_subtractor, max_width=960):
    small = resize_frame(frame, max_width)

    gray = to_grayscale(small)
    blurred = cv2.GaussianBlur(gray,(5,5),0)

    fg_mask = bg_subtractor.apply(blurred)

    # strong threshold → removes background noise
    _, binary = cv2.threshold(fg_mask,230,255,cv2.THRESH_BINARY)

    # denoise
    denoised = cv2.medianBlur(binary,7)

    opened = cv2.morphologyEx(denoised,cv2.MORPH_OPEN,_kernel_open)
    silhouette = cv2.morphologyEx(opened,cv2.MORPH_CLOSE,_kernel_close)

    return small, gray, fg_mask, silhouette