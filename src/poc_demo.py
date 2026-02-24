import cv2
import numpy as np
import matplotlib.pyplot as plt

# X = Input Video
video_path = "/Users/sukhmanikaur/Documents/codes/human_gait_analysis/human_gait_analysis/data/Walking (127).mp4"
cap = cv2.VideoCapture(video_path)

motion_values = []

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- f(x): Processing using OpenCV APIs ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion_value = np.sum(thresh)
    motion_values.append(motion_value)

    # Visualization (API usage)
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Motion Mask", thresh)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Y = Output Visualization ---
plt.plot(motion_values)
plt.title("Motion Signal Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Motion Intensity")
plt.show()
