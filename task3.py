import os
import cv2
import PIL
import numpy as np

left_mtx = np.load("left-mtx.npy")
left_dist = np.load("left-dist.npy")
right_mtx = np.load("right-mtx.npy")
right_dist = np.load("right-dist.npy")
fund_mtx = np.load("fundamental.npy")

# fund_mtx = [-0.0091700742,    1.7710310594,   -0.7461663948], [-1.3416247882,   -0.2328401698,   11.8163903279], [0.6604479052,  -11.7687990496,   -0.2732698516]
# fund_mtx = np.array(fund_mtx)

# fund_mtx = np.array([ [-0.0000004352,    0.0000841265,   -0.0615916207],
#   [-0.0000637634,   -0.0000110764 ,   0.6702695146],
#    [0.0504789932 ,  -0.6677701976 ,   1.0000000000]])

PATH = "my_images"
EPI_PATH = os.path.join(PATH, "Epipolar_Images")
# EPI_PATH = os.path.join(PATH, "Rectified Stereo Images")
epiList = os.listdir(EPI_PATH)
epiList.sort()

def plot_lines(param, img):
    y, x, z = img.shape
    a = param[0, 0]
    b = param[0, 1]
    c = param[0, 2]
    y1 = -a / b * 0 - c / b
    y2 = -a / b * x - c / b
    return int(y1), 0, int(y2), x


for file in epiList:
    img = cv2.imread(os.path.join(EPI_PATH, file))
    pattern = (10, 7)
    if "L" in file:
        left_img = img
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(left_mtx, left_dist, (w, h), 1, (w, h))
        left = cv2.undistort(img, left_mtx, left_dist, None, newcameramtx)
        ret, left_corners = cv2.findChessboardCorners(img, pattern)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        left_sub_corners = cv2.cornerSubPix(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY), np.float32(left_corners), (5, 5), (-1, -1), criteria)
        annotated_left = cv2.drawChessboardCorners(img, pattern, left_corners, ret)

    elif "R" in file:
        right_img = img
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(right_mtx, right_dist, (w, h), 1, (w, h))
        right = cv2.undistort(img, right_mtx, right_dist, None, newcameramtx)
        ret, right_corners = cv2.findChessboardCorners(img, pattern)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        right_sub_corners = cv2.cornerSubPix(cv2.cvtColor(right,cv2.COLOR_RGB2GRAY), np.float32(right_corners), (5, 5), (-1, -1), criteria)
        annotated_right = cv2.drawChessboardCorners(img, pattern, right_corners, ret)


left_pts = left_sub_corners
right_pts = right_sub_corners
left_lines = cv2.computeCorrespondEpilines(right_pts, 2, fund_mtx)
right_lines = cv2.computeCorrespondEpilines(left_pts, 1, fund_mtx)

for i in [0,10,20]:
    point = left_pts[i].astype(int)
    cv2.circle(left, (point[0, 0], point[0, 1]), 2, [255, 255, 0], thickness=-1)
    y1, x1, y2, x2 = plot_lines(left_lines[i], left)
    cv2.line(right, (x1, y1), (x2, y2), [0,255,255], thickness=2)

for i in [30, 40, 50]:
    point = right_pts[i].astype(int)
    cv2.circle(right, (point[0, 0], point[0, 1]), 2, [255, 255, 0], thickness=-1)
    y1, x1, y2, x2 = plot_lines(right_lines[i], right)
    cv2.line(left, (x1,y1), (x2,y2), [255, 0, 255], thickness=2)

cv2.imshow("left", left)
cv2.imwrite("undistort_epipoles_left.png", left)
cv2.waitKey()
cv2.imshow("right", right)
cv2.imwrite("undistort_epipoles_right.png", right)
cv2.waitKey()
cv2.destroyAllWindows()