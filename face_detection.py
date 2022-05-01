import cv2 as cv
import dlib
import numpy as np

image = cv.imread("eye.jpeg")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 1)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

predictor = dlib.shape_predictor('shape_68.dat')
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    for (x, y) in shape:
        cv.circle(image, (x, y), 2, (0, 0, 255), -1)

cv.imshow("img", image)
cv.waitKey(0)