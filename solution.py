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

# store all the good matches as per ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
        
print('stored good matches')

LENGTH_GOOD_MATCH = len(good)

if LENGTH_GOOD_MATCH > MIN_MATCH_COUNT:
    print("found enough matched - %d/%d" % (LENGTH_GOOD_MATCH, MIN_MATCH_COUNT))
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (LENGTH_GOOD_MATCH, MIN_MATCH_COUNT))
    matchesMask = None
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()