import numpy as np              # linear algebra
import cv2 as cv                #opencv
import matplotlib.pyplot as plt #plot images
import os
import math
import argparse


def find_features(small_image_path, starmap_path):
    small_img = cv.imread(small_image_path, cv.IMREAD_GRAYSCALE)       # queryImage
    starmap_img = cv.imread(starmap_path, cv.IMREAD_GRAYSCALE)          # trainImage

    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(small_img,None)
    kp2, des2 = sift.detectAndCompute(starmap_img,None)

    # Initiate BF matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
            
    length_good_match = len(good)

    if length_good_match > MIN_MATCH_COUNT:
        print("found enough matched - %d/%d" % (length_good_match, MIN_MATCH_COUNT))
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = small_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        angle = - math.atan2(M[1,0], M[0,0]) * 180 / math.pi
        print("Rotation angle ", angle)
        print("Corner points of small image onto StarMap ")
        print(np.int32(dst))

        starmap_img = cv.polylines(starmap_img,[np.int32(dst)],True,255,3, cv.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (length_good_match, MIN_MATCH_COUNT))
        matchesMask = None
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv.drawMatches(small_img,kp1,starmap_img,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--small-image-path",
        type=str,
        help="one of the small images"
    )

    parser.add_argument(
        "--starmap-path",
        type=str,
        help="the star map image"
    )

    args = parser.parse_args()
    small_image_path = args.small_image_path
    starmap_path = args.starmap_path

    if not small_image_path or not starmap_path:
        print("a small image and the starmap are required")
    else:
        find_features(small_image_path, starmap_path)
