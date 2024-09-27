import cv2 as cv
import numpy as np

img1 = cv.imread('../images/redtshirt.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('../images/redtshirtcut.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create(1000)

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print(des1.shape,des2.shape)

# Create a brute force matcher object wit default params
bf = cv.BFMatcher()
# Match descriptors.
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

print(len(good))
# Draw the good matches
result = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('ORB', result)
cv.waitKey(0)
cv.destroyAllWindows()