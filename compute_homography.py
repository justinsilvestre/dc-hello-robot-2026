# This script computes a homography matrix from image points to real-world floor coordinates.

import cv2
import numpy as np

img_pts = np.array([
    [251, 941],
    [326, 777],
    [387, 645],
    [439, 532],
    [481, 435],
    [518, 352],
    [550, 281],
    [576, 220],
    [439, 944],
    [494, 781],
    [542, 647],
    [581, 536],
    [613, 440],
    [641, 357],
    [665, 285],
    [685, 221],
    [625, 947],
    [664, 783],
    [698, 650],
    [722, 538],
    [745, 443],
    [762, 360],
    [778, 290],
    [793, 225],
    [812, 950],
    [832, 788],
    [848, 652],
    [862, 542],
    [874, 448],
    [884, 364],
    [892, 293],
    [897, 228],
    [995, 953],
    [995, 789],
    [998, 656],
    [1000, 546],
    [1001, 449],
    [1003, 369],
    [1003, 296],
    [1004, 232],
    [1178, 957],
    [1162, 794],
    [1149, 661],
    [1138, 548],
    [1130, 453],
    [1122, 370],
    [1116, 299],
    [1110, 234],
    [1363, 965],
    [1328, 800],
    [1302, 665],
    [1278, 551],
    [1259, 454],
    [1242, 372],
    [1228, 298],
    [1215, 236],
], dtype=np.float32)


world_pts_steps = np.array([
    [0, 0],  [1, 0],  [2, 0],  [3, 0],  [4, 0],  [5, 0],  [6, 0],  [7, 0],
    [0, 1],  [1, 1],  [2, 1],  [3, 1],  [4, 1],  [5, 1],  [6, 1],  [7, 1],
    [0, 2],  [1, 2],  [2, 2],  [3, 2],  [4, 2],  [5, 2],  [6, 2],  [7, 2],
    [0, 3],  [1, 3],  [2, 3],  [3, 3],  [4, 3],  [5, 3],  [6, 3],  [7, 3],
    [0, 4],  [1, 4],  [2, 4],  [3, 4],  [4, 4],  [5, 4],  [6, 4],  [7, 4],
    [0, 5],  [1, 5],  [2, 5],  [3, 5],  [4, 5],  [5, 5],  [6, 5],  [7, 5],
    [0, 6],  [1, 6],  [2, 6],  [3, 6],  [4, 6],  [5, 6],  [6, 6],  [7, 6]
], dtype=np.float32)

world_pts = world_pts_steps * 0.5  # Convert steps to meters

assert img_pts.shape == world_pts.shape and img_pts.shape[0] >= 4

H, inliers = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC)
if H is None:
    raise RuntimeError("Homography failed. Check point order / non-collinearity.")

np.save("floor_homography.npy", H)

print("Saved floor_homography.npy")
print("H =\n", H)
print("Inliers:", int(inliers.sum()), "/", len(inliers))
