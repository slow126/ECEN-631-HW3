import os
import cv2
import PIL
import numpy as np


PATH = "my_images"
if PATH == "practice_images":
    scale = 2
else:
    scale = 3.88

LEFT_PATH = os.path.join(PATH, "Left Camera Images")
leftList = os.listdir(LEFT_PATH)
leftList.sort()
RIGHT_PATH = os.path.join(PATH, "Right Camera Images")
rightList = os.listdir(RIGHT_PATH)
rightList.sort()
STEREO_PATH = os.path.join(PATH, "Stereo Images")
stereoList = os.listdir(STEREO_PATH)


def find_corners(in_img, pattern=(10, 7)):

    color_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2RGB)
    # gray_gpu = cv2.UMat(gray)
    # pattern = (10, 7)
    pattern2 = (23, 11)

    objp = np.zeros((pattern[1] * pattern[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    ret, corners = cv2.findChessboardCorners(color_img, pattern)

    annotated_gray = color_img
    sub_corners = None

    if ret is True:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        sub_corners = cv2.cornerSubPix(in_img, np.float32(corners), (5, 5), (-1, -1), criteria)
        imgpoints.append(np.squeeze(sub_corners))
        annotated_gray = cv2.drawChessboardCorners(color_img, pattern, sub_corners, ret)

    if ret is False:
        ret2, corners = cv2.findChessboardCorners(in_img, pattern2)

        if ret2 is True:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            sub_corners = cv2.cornerSubPix(in_img, np.float32(corners), (5, 5), (-1, -1), criteria)
            annotated_gray = cv2.drawChessboardCorners(color_img, pattern2, sub_corners, ret)
            annotated_gray = cv2.cvtColor(annotated_gray, cv2.COLOR_GRAY2RGB)

    # cv2.imshow("img", annotated_gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return annotated_gray, sub_corners, objpoints, imgpoints


def process_list(PATH, imgList, camera):
    obj_vect, img_vect = [], []
    for file in imgList:
        img = cv2.imread(os.path.join(PATH, file))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        annotated_img, corners, obj_pt, img_pt = find_corners(gray, pattern=(10, 7))
        obj_vect += [obj_pt[0]]
        img_vect += [img_pt[0]]

    obj_vect = scale * np.array(obj_vect)
    img_vect = np.array(img_vect)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_vect, img_vect, gray.shape[::-1], None, None) #flags=cv2.CALIB_RATIONAL_MODEL)
    np.save(camera + "-mtx.npy", mtx)
    np.save(camera + "-dist.npy", dist)
    np.save(camera + "-rvecs.npy", rvecs)
    np.save(camera + "-tvecs.npy", tvecs)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print(roi)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    ann = annotated_img
    # dst = dst[y:y + h, x:x + w]
    # ann = annotated_img[y:y+h, x:x+h]
    cv2.imwrite('calibresult.png', dst)
    calibrated_img = np.concatenate((ann, dst), axis=1)
    # cv2.imshow("calibraton", dst)
    # cv2.imshow("Calibration", calibrated_img)
    cv2.imwrite(camera + '-side_by_side.jpg', calibrated_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


process_list(LEFT_PATH, leftList, "left")
process_list(RIGHT_PATH, rightList, "right")
