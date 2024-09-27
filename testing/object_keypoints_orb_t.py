import cv2 as cv
import numpy as np

img1 = cv.imread('../images/jwellary.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('../images/jwellaryring.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create(1000)

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# print(des1.shape)

# Draw the keypounts location, not size and orientation
result = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
cv.imshow('ORB', result)

cv.waitKey(0)
cv.destroyAllWindows()