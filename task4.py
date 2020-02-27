import os
import cv2
import PIL
import numpy as np

def plot_lines(param, img):
    y, x, z = img.shape
    a = param[0, 0]
    b = param[0, 1]
    c = param[0, 2]
    y1 = -a / b * 0 - c / b
    y2 = -a / b * x - c / b
    return int(y1), 0, int(y2), x

left_mtx = np.load("left-mtx.npy")
left_dist = np.load("left-dist.npy")
right_mtx = np.load("right-mtx.npy")
right_dist = np.load("right-dist.npy")

R = np.load("R.npy")
T = np.load("T.npy")
E = np.load("E.npy")
F = np.load("F.npy")

PATH = "my_images"
if PATH == "practice_images":
    scale = 2
else:
    scale = 3.88

STEREO_PATH = os.path.join(PATH, "Unectified Stereo Images")
stereoList = os.listdir(STEREO_PATH)
stereoList.sort()
pattern = (10, 7)

for file in stereoList:
    img = cv2.imread(os.path.join(STEREO_PATH, file))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(left_mtx, left_dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, left_mtx, left_dist, None, newcameramtx)
    if "L" in file:
        left = img
        old_left = img

    if "R" in file:
        right = cv2.imread(os.path.join(STEREO_PATH, file))
        old_right = img


y, x, z = left.shape
R1, R2, P1, P2, Q, SIZE1, SIZE2 = cv2.stereoRectify(left_mtx, left_dist, right_mtx, right_dist, (x,y), R, T)

l_map1, l_map2 = cv2.initUndistortRectifyMap(left_mtx, left_dist, R1, P1, (x,y), cv2.CV_32FC1)
left_undist = cv2.remap(left, l_map1, l_map2, cv2.INTER_LINEAR)
cv2.imshow("Left Undistort", left_undist)
cv2.imwrite("my_images/Rectified Stereo Images/L1.png", left_undist)
cv2.waitKey()

r_map1, r_map2 = cv2.initUndistortRectifyMap(right_mtx, right_dist, R2, P2, (x,y), cv2.CV_32FC1)
right_undisort = cv2.remap(right, r_map1, r_map2, cv2.INTER_LINEAR)
cv2.imshow("Right Undisort", right_undisort)
cv2.imwrite("my_images/Rectified Stereo Images/R1.png", right_undisort)
cv2.waitKey()

left_diff = cv2.absdiff(old_left, left_undist)
cv2.imshow("Left Diff", left_diff)
cv2.imwrite("Rectified_Left_diff.png", left_diff)
cv2.waitKey()
right_diff = cv2.absdiff(old_right, right_undisort)
cv2.imshow("Right Diff", right_diff)
cv2.imwrite("Rectified_Right_diff.png", right_diff)
cv2.waitKey()

ret, left_corners = cv2.findChessboardCorners(left_undist, pattern)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
left_sub_corners = cv2.cornerSubPix(cv2.cvtColor(left_undist, cv2.COLOR_RGB2GRAY), np.float32(left_corners), (5, 5),
                                            (-1, -1), criteria)

ret, right_corners = cv2.findChessboardCorners(right_undisort, pattern)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
right_sub_corners = cv2.cornerSubPix(cv2.cvtColor(right_undisort, cv2.COLOR_RGB2GRAY), np.float32(right_corners), (5, 5),
                                             (-1, -1), criteria)

left_pts = left_sub_corners
right_pts = right_sub_corners
left_lines = cv2.computeCorrespondEpilines(right_pts, 2, F)
right_lines = cv2.computeCorrespondEpilines(left_pts, 1, F)

for i in [0,10,20]:
    point = left_pts[i].astype(int)
    cv2.circle(left_undist, (point[0, 0], point[0, 1]), 2, [255, 255, 0], thickness=-1)
    y1, x1, y2, x2 = plot_lines(left_lines[i], left_undist)
    cv2.line(right_undisort, (x1, y1), (x2, y2), [0,255,255], thickness=2)

for i in [30, 40, 50]:
    point = right_pts[i].astype(int)
    cv2.circle(right_undisort, (point[0, 0], point[0, 1]), 2, [255, 255, 0], thickness=-1)
    y1, x1, y2, x2 = plot_lines(right_lines[i], right_undisort)
    cv2.line(left_undist, (x1,y1), (x2,y2), [255, 0, 255], thickness=2)

cv2.imshow("left", left_undist)
cv2.imwrite("rectified_epipoles_left.png", left_undist)
cv2.waitKey()
cv2.imshow("right", right_undisort)
cv2.imwrite("rectified_epipoles_right.png", right_undisort)
cv2.waitKey()
cv2.destroyAllWindows()
