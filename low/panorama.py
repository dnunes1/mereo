# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:41:16 2013

@author: danilonunes
"""
import cv2
import numpy as np


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs


def panorama(img1, img2):
    '''
    img1 left image (RGB)
    img2 right image (RGB)
    result (output) panorama image
    '''

    gray_image1 = cv2.cvtColor(img1, cv2.cv.CV_RGB2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.cv.CV_RGB2GRAY)

#   detector = cv2.FeatureDetector_create("SURF")
    detector = cv2.SURF(400)
    kp1 = detector.detect(gray_image1)
    kp2 = detector.detect(gray_image2)
    print 'keypoints in image1: %d, image2: %d' % (len(kp1), len(kp2))

    descriptor = cv2.DescriptorExtractor_create("SURF")
    k1, d1 = descriptor.compute(gray_image1, kp1)  # keypoints object
    k2, d2 = descriptor.compute(gray_image2, kp2)  # keypoints scene

    # match the keypoints
    matcher = cv2.DescriptorMatcher_create("FlannBased")
    matches = matcher.knnMatch(d1, d2, k=2)
    print 'number of matches: %d' % len(matches)

    # dist = [m.distance for m in matches]
    # min_dist = np.min(dist)
    # max_dist = np.max(dist)
    # good_matches = [m for m in matches if m.distance < (3 * min_dist)]
    good_matches = filter_matches(k1, k2, matches, ratio=0.75)
    print 'number of matches higher than threshold: %d' % len(good_matches)

    obj = np.array([k1[m.queryIdx].pt for m in good_matches])
    scn = np.array([k2[m.trainIdx].pt for m in good_matches])

    homography_matrix = cv2.findHomography(obj, scn, cv2.cv.CV_RANSAC)
    r1, c1 = gray_image1.shape
    r2, c2 = gray_image2.shape
    result = cv2.warpPerspective(img1, homography_matrix[0], (c1 + c2, r1))
    result[:r2, :c2, :] = img2

    return result
