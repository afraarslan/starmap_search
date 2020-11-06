import numpy as np  # linear algebra
import cv2 as cv  # opencv
import matplotlib.pyplot as plt  # plot images
import math
import argparse

MIN_MATCH_COUNT = 10

def locate_area(small_image_path, starmap_path):
    keypoints1, keypoints2, matches, small_img, starmap_img = feature_matching(small_image_path, starmap_path)

    good, length_good_match = find_good_matches(matches)

    if length_good_match > MIN_MATCH_COUNT:
        print("found enough matched - %d/%d" % (length_good_match, MIN_MATCH_COUNT))
        matchesMask, starmap_img = find_corner_points(good, keypoints1, keypoints2, small_img, starmap_img)

    else:
        print("Not enough matches are found - %d/%d" % (length_good_match, MIN_MATCH_COUNT))
        matchesMask = None

    draw_final_image(good, keypoints1, keypoints2, matchesMask, small_img, starmap_img)


def draw_final_image(good, keypoints1, keypoints2, matchesMask, small_img, starmap_img):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    final_img = cv.drawMatches(small_img, keypoints1, starmap_img, keypoints2, good, None, **draw_params)
    plt.imshow(final_img, 'gray'), plt.show()


def find_corner_points(good, keypoints1, keypoints2, small_img, starmap_img):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = small_img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    angle = math.atan2(M[1, 0], M[0, 0])
    print("Rotation angle ", angle)
    print("Corner points of small image onto StarMap ")
    print(np.int32(dst))
    starmap_img = cv.polylines(starmap_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    return matchesMask, starmap_img


def find_good_matches(matches):
    # store all the good matches as per ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    length_good_match = len(good)
    return good, length_good_match


def feature_matching(small_image_path, starmap_path):
    small_img = cv.imread(small_image_path, cv.IMREAD_GRAYSCALE)
    starmap_img = cv.imread(starmap_path, cv.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(small_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(starmap_img, None)

    # Initiate BF matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    return keypoints1, keypoints2, matches, small_img, starmap_img


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
        locate_area(small_image_path, starmap_path)
