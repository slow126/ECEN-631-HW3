import os
import cv2
import PIL
import numpy as np

left_mtx = np.load("left-mtx.npy")
left_dist = np.load("left-dist.npy")
right_mtx = np.load("right-mtx.npy")
right_dist = np.load("right-dist.npy")

true_left_mtx = np.array([[1153.7918353122,    0.0000000000,  311.6495082713],
   [0.0000000000, 1152.7061368496,  247.7409370695],
   [0.0000000000,    0.0000000000,    1.0000000000]])
true_left_dist = np.array([ -0.2574338164,
   0.3395576609,
   0.0011179409,
  -0.0002030712,
  -0.5947353243])
true_right_mtx = np.array([[1149.6505965772,    0.0000000000,  326.3569432986],
   [0.0000000000, 1148.0218738819,  224.6062742604],
   [0.0000000000,    0.0000000000,    1.0000000000]])
true_right_dist = np.array([  -0.2950621013,
   1.1296741454,
  -0.0010482716,
  -0.0014052463,
  -9.9589684633])

PATH = "my_images"
if PATH == "practice_images":
    scale = 2
else:
    scale = 3.88

STEREO_PATH = os.path.join(PATH, "Stereo Images")
stereoList = os.listdir(STEREO_PATH)
stereoList.sort()


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

    # cv2.imshow("img", annotated_gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return annotated_gray, sub_corners, objpoints, imgpoints


def process_list(PATH, imgList, camera):
    obj_vect, left_img_vect = [], []
    right_img_vect = []
    for file in imgList:
        if "L" in file:
            img = cv2.imread(os.path.join(PATH, file))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            annotated_img, corners, obj_pt, left_img_pt = find_corners(gray, pattern=(10, 7))
            obj_vect += [obj_pt[0]]
            left_img_vect += [left_img_pt[0]]
        elif "R" in file:
            img = cv2.imread(os.path.join(PATH, file))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            annotated_img, corners, obj_pt, right_img_pt = find_corners(gray, pattern=(10, 7))
            # obj_vect += [obj_pt[0]]
            right_img_vect += [right_img_pt[0]]

    obj_vect = np.array(obj_vect)
    left_img_vect = np.array(left_img_vect)
    right_img_vect = np.array(right_img_vect)
    y, x, z = img.shape
    obj_vect = scale * obj_vect

    # cv2.stereoCalibrate(obj_vect, left_img_vect, right_img_vect, left_mtx, left_dist, right_mtx, right_dist, (x, y), R=None, T=None, E=None, F=None, flags=cv2.CALIB_FIX_INTRINSIC)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.000001)
    ret, left_mtx_out, left_dist_out, right_mtx_out, right_dist_out, R, T, E, F = cv2.stereoCalibrate(objectPoints=obj_vect, imagePoints1=left_img_vect, imagePoints2=right_img_vect, cameraMatrix1=left_mtx, distCoeffs1=left_dist, cameraMatrix2=right_mtx, distCoeffs2=right_dist, imageSize=(x, y), flags=cv2.CALIB_FIX_INTRINSIC)
    # ret, left_mtx_out, left_dist_out, right_mtx_out, right_dist_out, R, T, E, F = cv2.stereoCalibrate(obj_vect, left_img_vect, right_img_vect, true_left_mtx, true_left_dist, true_right_mtx, true_right_dist, (y, x), flags=cv2.CALIB_FIX_INTRINSIC)
    np.save("fundamental.npy", F)
    print(R)
    print(T)
    print(E)
    print(F)


process_list(STEREO_PATH, stereoList, "stereo")
