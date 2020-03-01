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

# PATH = "practice_images"
PATH = "my_images"
if PATH == "practice_images":
    scale = 2
else:
    scale = 3.88

STEREO_PATH = os.path.join(PATH, "Unrectified Stereo Images")
stereoList = os.listdir(STEREO_PATH)
stereoList.sort()
pattern = (10, 7)

for file in stereoList:
    img = cv2.imread(os.path.join(STEREO_PATH, file))
    h, w = img.shape[:2]
    if "L" in file:
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(left_mtx, left_dist, (w, h), 1, (w, h))
        # img = cv2.undistort(img, left_mtx, left_dist, None, newcameramtx)
        left = img
        old_left = img

    if "R" in file:
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(right_mtx, right_dist, (w, h), 1, (w, h))
        # img = cv2.undistort(img, right_mtx, right_dist, None, newcameramtx)
        right = cv2.imread(os.path.join(STEREO_PATH, file))
        old_right = img


ret, left_corners = cv2.findChessboardCorners(left, pattern)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
left_sub_corners = cv2.cornerSubPix(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), np.float32(left_corners), (5, 5),
                                            (-1, -1), criteria)

ret, right_corners = cv2.findChessboardCorners(right, pattern)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
right_sub_corners = cv2.cornerSubPix(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), np.float32(right_corners), (5, 5),
                                             (-1, -1), criteria)

left_pts = left_sub_corners
right_pts = right_sub_corners
left_lines = cv2.computeCorrespondEpilines(right_pts, 2, F)
right_lines = cv2.computeCorrespondEpilines(left_pts, 1, F)

for i in [0,10,20]:
    point = left_pts[i].astype(int)
    cv2.circle(left, (point[0, 0], point[0, 1]), 2, [255, 255, 0], thickness=-1)
    y1, x1, y2, x2 = plot_lines(left_lines[i], left)
    cv2.line(right, (x1, y1), (x2, y2), [0,255,255], thickness=2)

for i in [0, 10, 20]:
    point = right_pts[i].astype(int)
    cv2.circle(right, (point[0, 0], point[0, 1]), 2, [255, 255, 0], thickness=-1)
    y1, x1, y2, x2 = plot_lines(right_lines[i], right)
    cv2.line(left, (x1,y1), (x2,y2), [255, 0, 255], thickness=2)

# cv2.imshow("left", left)
# cv2.imwrite("rectified_epipoles_left.png", left)
# cv2.waitKey()
# cv2.imshow("right", right)
# cv2.imwrite("rectified_epipoles_right.png", right)
# cv2.waitKey()
# cv2.destroyAllWindows()

y, x, z = left.shape
R1, R2, P1, P2, Q, SIZE1, SIZE2 = cv2.stereoRectify(left_mtx, left_dist, right_mtx, right_dist, (x,y), R, T)

original_side = np.concatenate((left, right), axis=1)
cv2.imshow("Original sid-by-side", original_side)
cv2.imwrite("original_side.png", original_side)
# cv2.imshow("Left Original", old_left)
# cv2.imwrite("task4_left_original.png", old_left)
# cv2.imshow("Right Original", old_right)
# cv2.imwrite("task4_right_original.png", old_right)

l_map1, l_map2 = cv2.initUndistortRectifyMap(left_mtx, left_dist, R1, P1, (x,y), cv2.CV_32FC1)
left_undist = cv2.remap(left, l_map1, l_map2, cv2.INTER_LINEAR)
# cv2.imshow("Left Undistort", left_undist)
# cv2.imwrite("my_images/Rectified Stereo Images/L1.png", left_undist)
# cv2.waitKey()

r_map1, r_map2 = cv2.initUndistortRectifyMap(right_mtx, right_dist, R2, P2, (x,y), cv2.CV_32FC1)
right_undisort = cv2.remap(right, r_map1, r_map2, cv2.INTER_LINEAR)
# cv2.imshow("Right Undisort", right_undisort)
# cv2.imwrite("my_images/Rectified Stereo Images/R1.png", right_undisort)
# cv2.waitKey()

rect_side = np.concatenate((left_undist, right_undisort), axis=1)
cv2.imshow("Rectified side-by-side", rect_side)
cv2.imwrite("rect_side.png", rect_side)

left_diff = cv2.absdiff(old_left, left_undist)
# cv2.imshow("Left Diff", left_diff)
# cv2.imwrite("Rectified_Left_diff.png", left_diff)
# cv2.waitKey()
right_diff = cv2.absdiff(old_right, right_undisort)
# cv2.imshow("Right Diff", right_diff)
# cv2.imwrite("Rectified_Right_diff.png", right_diff)
# cv2.waitKey()
diff_side = np.concatenate((left_diff, right_diff), axis=1)
cv2.imshow("Difference side-by-side", diff_side)
cv2.imwrite("diff_side.png", diff_side)
cv2.waitKey()


# cv2.imshow("left", left_undist)
# cv2.imwrite("rectified_epipoles_left.png", left_undist)
# cv2.waitKey()
# cv2.imshow("right", right_undisort)
# cv2.imwrite("rectified_epipoles_right.png", right_undisort)
# cv2.waitKey()
cv2.destroyAllWindows()
