# Calibrate camera by clicking floor points in the camera image.

import cv2
import numpy as np
from datetime import datetime

CAM_INDEX = 0
FLIP = False
N_POINTS = 8  
OUT_PATH = None  

clicked = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append([x, y])
        print(f"clicked {len(clicked)}: ({x}, {y})")

def draw_points(frame, pts):
    vis = frame.copy()
    for i, (u, v) in enumerate(pts):
        cv2.circle(vis, (u, v), 6, (0, 255, 255), -1)
        cv2.putText(
            vis, str(i+1), (u+8, v-8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
    return vis

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read from camera.")

    if FLIP:
        frame = cv2.flip(frame, 1)

    cv2.namedWindow("calib")
    cv2.setMouseCallback("calib", on_mouse)

    print(f"Click {N_POINTS} floor points (X marks).")
    print("Press ENTER when done, ESC to quit.")

    while True:
        vis = draw_points(frame, clicked)
        cv2.imshow("calib", vis)

        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # ESC
            break

        if k == 13:  # ENTER
            if len(clicked) < N_POINTS:
                print(f"Need {N_POINTS} points, currently have {len(clicked)}")
            else:
                # Save annotated image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = OUT_PATH or f"calib_clicked_{N_POINTS}pts_{timestamp}.jpg"
                final_vis = draw_points(frame, clicked)
                ok_save = cv2.imwrite(out_path, final_vis)
                if ok_save:
                    print(f"\nSaved annotated calibration image: {out_path}")
                else:
                    print("\nWARNING: Failed to save image (cv2.imwrite returned False).")

                break

    cv2.destroyAllWindows()

    if len(clicked) >= N_POINTS:
        print("\nPaste this into your homography code:\n")
        print("img_pts = np.array([")
        for (u, v) in clicked:
            print(f"    [{u}, {v}],")
        print("], dtype=np.float32)")

if __name__ == "__main__":
    main()
