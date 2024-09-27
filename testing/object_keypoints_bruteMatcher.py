import cv2 as cv
import numpy as np

img1 = cv.imread('../images/womenJacket.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('../images/jwellaryring.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create(1000)

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print(des1.shape,des2.shape)

# Create a brute force matcher object.
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
print(len(matches))
# Draw first 10 matches.
result = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('ORB', result)
cv.waitKey(0)
cv.destroyAllWindows()