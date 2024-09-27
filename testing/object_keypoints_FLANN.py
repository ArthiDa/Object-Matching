import cv2 as cv
import numpy as np

img1 = cv.imread('../images/f805-01-500x500.webp', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('../images/aula-f810-honeycomb-gaming-mouse-01-500x500.webp', cv.IMREAD_GRAYSCALE)

img1 = cv.resize(img1, (520,520))
img2 = cv.resize(img2, (520,520))

# Initiate ORB detector
orb = cv.ORB_create(1000)
# sift = cv.SIFT_create(1000)

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print(des1.shape, type(des2))
# FLANN -> FAST Library for Approximate Nearest Neighbors
# LSH -> Locality Sensitive Hashing
# table_number=6 -> This sets the number of hash tables to be used for the LSH index to 6.
# key_size=12 -> This sets the size of the hash key in the LSH index to 12.
# multi_probe_level=1 -> This sets the number of bits to shift to check the number of hash collisions in the LSH index to 1.
# FLANN parameters 
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1) #12 20 2
# Sets the number of times the algorithm will traverse the LSH table to perform the approximate nearest neighbor search
search_params = dict(checks=50) # or pass empty dictionary

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)


# Draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

cn = 0
print(len(matches))
# ratio test as per Lowe's paper
for i, match in enumerate(matches):
    if len(match) == 2:
        (m, n) = match
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1, 0]
            cn += 1
    else:
        pass
print("Total matches: ", len(matches), "Good matches: ", cn)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)

result = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
cv.imshow('ORB', result)
cv.waitKey(0)
cv.destroyAllWindows()
