import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------------
# X = INPUT VIDEO
# -------------------------------
video_path = "/Users/sukhmanikaur/Documents/codes/human_gait_analysis/human_gait_analysis/data/CV_dataset/demo.mp4"
cap = cv2.VideoCapture(video_path)

motion_values = []
centroids = []
symmetry_values = []

ret, prev_frame = cap.read()
if not ret:
    print("Video not found.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# -------------------------------
# PROCESS VIDEO
# f(x) = Classical CV APIs
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Motion extraction
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion_value = np.sum(thresh)
    motion_values.append(motion_value)

    # -------------------------------
    # Centroid computation
    # -------------------------------
    ys, xs = np.where(thresh > 0)
    if len(xs) > 0:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
    else:
        cx, cy = 0, 0

    centroids.append((cx, cy))

    # Draw centroid
    if cx != 0 and cy != 0:
        cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)

    # -------------------------------
    # Symmetry calculation
    # -------------------------------
    h, w = thresh.shape
    left_motion = np.sum(thresh[:, :w//2])
    right_motion = np.sum(thresh[:, w//2:])
    symmetry_values.append(abs(left_motion - right_motion))

    # -------------------------------
    # Motion overlay visualization
    # -------------------------------
    overlay = frame.copy()
    overlay[thresh > 0] = [0,0,255]

    # Bounding boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x,y,wc,hc = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+wc,y+hc),(0,255,0),2)

    # Show outputs
    cv2.imshow("Original + Bounding Boxes", frame)
    cv2.imshow("Motion Mask", thresh)
    cv2.imshow("Motion Overlay", overlay)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# Y = GAIT METRICS COMPUTATION
# -------------------------------
motion_values = np.array(motion_values)
centroids = np.array(centroids)

# Walking speed (pixel/frame)
dx = np.diff(centroids[:,0])
walking_speed = np.mean(np.abs(dx))

# Step frequency using motion peaks
peaks, _ = find_peaks(motion_values, distance=10)
step_frequency = len(peaks) / len(motion_values)

# Approx stride length
if len(peaks) > 1:
    stride_lengths = np.abs(np.diff(centroids[peaks][:,0]))
    avg_stride = np.mean(stride_lengths)
else:
    avg_stride = 0

# Symmetry score
symmetry_score = np.mean(symmetry_values)

# Motion smoothness
motion_diff = np.diff(motion_values)
motion_smoothness = np.std(motion_diff)

# -------------------------------
# PRINT METRICS
# -------------------------------
print("\n---- GAIT METRICS ----")
print("Walking Speed (pixel/frame):", walking_speed)
print("Step Frequency:", step_frequency)
print("Approx Stride Length:", avg_stride)
print("Symmetry Score:", symmetry_score)
print("Motion Smoothness:", motion_smoothness)

# -------------------------------
# VISUALIZE MOTION GRAPH
# -------------------------------
window = 10
smoothed = np.convolve(motion_values, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,4))
plt.plot(smoothed)
plt.title("Smoothed Motion Signal Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Motion Intensity")
plt.show()
