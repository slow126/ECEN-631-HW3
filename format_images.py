import os
import cv2
import PIL
import numpy as np


PATH = "my_images"
if PATH == "practice_images":
    scale = 2
else:
    scale = 3.88

RIGHT_PATH = os.path.join(PATH, "Right Camera Images")
rightList = os.listdir(RIGHT_PATH)
rightList.sort()

count = 1
for file in rightList:
    img = cv2.imread(os.path.join(RIGHT_PATH, file))
    cv2.imwrite("R" + str(count) + ".bmp", img)
