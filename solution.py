import numpy as np              # linear algebra
import cv2 as cv                #opencv
import matplotlib.pyplot as plt #plot images
import os


img1 = cv.imread('Small_area.png', cv.IMREAD_GRAYSCALE)       # queryImage
img2 = cv.imread('StarMap.png', cv.IMREAD_GRAYSCALE)          # trainImage

MIN_MATCH_COUNT = 10

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print('detected and computed keypoints')

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
        
print('stored good matches')
