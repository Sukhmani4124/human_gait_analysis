import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from video_loader import load_video,get_video_properties,extract_frames,release_video
from preprocessing import create_background_subtractor,preprocess_frame


def main(video_path: str):

    cap = load_video(video_path)
    props = get_video_properties(cap)

    delay = max(1,int(1000/props["fps"]))

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Total frames: {props['frame_count']}")

    bg_subtractor = create_background_subtractor()

    # let background model learn first frames
    for _ in range(40):
        cap.read()

    for _, frame in extract_frames(cap):

        small, gray, mask, silhouette = preprocess_frame(
            frame, bg_subtractor, max_width=960
        )

        # ⭐ ALL 4 PREPROCESSING WINDOWS
        cv2.imshow("Original", small)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("FG Mask", mask)
        cv2.imshow("Silhouette", silhouette)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    release_video(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1])