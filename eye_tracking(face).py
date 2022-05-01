from gaze_tracking import GazeTracking
import cv2

image = cv2.imread("test.jpg")

tracker = GazeTracking()

tracker.refresh(image)

new_image = tracker.annotated_frame()

cv2.imshow("img", new_image)
cv2.waitKey(0)