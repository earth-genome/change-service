""" test_bfmatch.py
Compare BFMatch.match() with built-in crosscheck and .kNNmatch() with manual crosscheck.
:params:
: image_file1: First image
: image_file2: Second image
"""

import argparse
import numpy as np
import cv2
import os

# global parameters
FEATURES = ("SIFT", "KAZE")[0]              # choose feature type
KAZE_PARAMETER = 0.0003                     # empirical
MATCH_PROXIMITY_IN_PIXELS = 4               # empirical
K_NEAREST = 1                               # empirical


def _are_close(kpa,kpb,distance):
	# Returns true if keypoints are separated by less than distance

    a_coords = np.array([kpa.pt[0], kpa.pt[1]])
    b_coords = np.array([kpb.pt[0], kpb.pt[1]])
    dist = np.linalg.norm(a_coords - b_coords)
    if dist > 0 and dist < distance:
        return True
    else:
        return False

def do_keypoint_detection(im1,im2,feature_detector):
    # Returns keypoints and descriptors for selected feature type

    kps1,desc1 = feature_detector.detectAndCompute(im1,None)
    kps2,desc2 = feature_detector.detectAndCompute(im2,None)

    return kps1,kps2,desc1,desc2

def do_non_knn_match(kps1,kps2,desc1,desc2):
    # Do matching with standard BFMatch.match() and built-in cross-check

    match_candidates = BFMATCH_CC.match(desc1,desc2)
    """
    matches = [
        m for m in match_candidates if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS)
    ]
    """
    matches = match_candidates

    return matches

def do_knn_match(kps1,kps2,desc1,desc2):
    # Do matching with kNN search

    match_candidates = BFMATCH.knnMatch(desc1,desc2,k=K_NEAREST)
    matches_forward = []
    #n_not_first_match_forward = 0
    for knnlist in match_candidates:
        matches_forward.append(knnlist[0])      # ignore proximity test for now
    """
        for n,m in enumerate(knnlist):
            if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS):
                matches_forward.append(m)
                if n > 0:
                    n_not_first_match_forward += 1
                break   # only pick first match that passes proximity test (closest in descriptor space)

    n_forward = len(matches_forward)
    """

    match_candidates = BFMATCH.knnMatch(desc2,desc1,k=K_NEAREST)
    matches_backward = []
    #n_not_first_match_backward = 0
    for knnlist in match_candidates:
        matches_backward.append(knnlist[0])
    """
        for n,m in enumerate(knnlist):
            if _are_close(kps1[m.trainIdx],kps2[m.queryIdx],MATCH_PROXIMITY_IN_PIXELS):
                matches_backward.append(m)
                if n > 0:
                    n_not_first_match_backward += 1
                break   # only pick first match that passes proximity test (closest in descriptor space)

    n_backward = len(matches_backward)
    """

    # cross-check forward and backward match lists
    matches = []
    for mf in matches_forward:
        for mb in matches_backward:
            if mf.trainIdx == mb.queryIdx and mf.queryIdx == mb.trainIdx:
            #if mf.trainIdx == mb.queryIdx:    # incorrect test - need to check both ways
                matches.append(mf)
                matches_backward.remove(mb)    # unique matching only
                break

    #return matches, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward
    return matches


# read args
parser = argparse.ArgumentParser()
parser.add_argument('image_file1', type=str)
parser.add_argument('image_file2', type=str)
args = parser.parse_args()

# instantiate global OpenCV objects
if FEATURES == "SIFT":
    SIFT = cv2.xfeatures2d.SIFT_create()
    feature_detector = SIFT       
elif FEATURES == "KAZE":
    KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
    feature_detector = KAZE

BFMATCH = cv2.BFMatcher()       
BFMATCH_CC = cv2.BFMatcher(crossCheck=True)

# load image files
im1 = cv2.imread(args.image_file1,0)
im2 = cv2.imread(args.image_file2,0)

# detect keypoints and descriptors
kps1,kps2,desc1,desc2 = do_keypoint_detection(im1,im2,feature_detector)

# find matches with built-in cross-check
matches_cc = do_non_knn_match(kps1,kps2,desc1,desc2)

# find matches
#matches_knn, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward = do_knn_match(kps1,kps2,desc1,desc2)
matches_knn = do_knn_match(kps1,kps2,desc1,desc2)

# report
print("Built-in CC matches: {0}; Manual CC matches: {1}".format(len(matches_cc),len(matches_knn)))

#matches_cc_set = set(matches_cc)
#matches_knn_set = set(matches_knn)
#matches_cc_only = matches_cc_set.difference(matches_knn_set)
#matches_knn_only = matches_knn_set.difference(matches_cc_set)
#print("Matches only in CC: {0}; Matches only in kNN: {1}".format(len(matches_cc_only),len(matches_knn_only)))

print("Finished")
