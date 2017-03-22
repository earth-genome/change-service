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
import pdb

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

def do_non_knn_match(kps1,kps2,desc1,desc2,CV2crosscheck,ourXcheck):
    # Do matching with BFMatch.match(); with or without crosscheck, with or without proximity test

    if CV2crosscheck:
        match_candidates = BFMATCH_CC.match(desc1,desc2)
        print 'CV2Xcheck len match-cands forward: {}'.format(len(match_candidates))
        match_cand_back = BFMATCH_CC.match(desc2,desc1)
        print 'CV2Xcheck len match-cands backward: {}'.format(len(match_cand_back))
    else:
        match_candidates = BFMATCH.match(desc1,desc2)
        print 'match cand forward: {}'.format(len(match_candidates))

    #matches_forward = [m for m in match_candidates
    #                   if _are_close(kps1[m.queryIdx],
    #                        kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS)]
    matches_forward = list(match_candidates)
    
    if ourXcheck:
        match_candidates_backward = BFMATCH.match(desc2,desc1)
        print 'match cand backw: {}'.format(len(match_candidates_backward))
        #matches_backward = [m for m in match_candidates_backward
        #                     if _are_close(kps2[m.queryIdx],
        #                    kps1[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS)]
        matches_backward = list(match_candidates_backward)
        matches = []

        print 'mf: {}, mb: {}'.format(len(matches_forward),
                                      len(matches_backward))
        ct = 0
        noct = 0
        failct = 0
        for mf in matches_forward:
            candidates = [mb for mb in matches_backward
                          if mf.trainIdx==mb.queryIdx]
            if len(candidates) == 0:
                noct +=1
            if len(candidates) > 1:
                print candidates
            for c in candidates:
                if c.trainIdx==mf.queryIdx:
                    matches.append(mf)
                    ct +=1
                    #break
                else:
                    failct += 1
        print 'count: {}'.format(ct)
        print 'noct: {}'.format(noct)
        print 'failct: {}'.format(failct)
        print 'matches: {}'.format(len(matches))
        #for mf in matches_forward:
        #    for mb in matches_backward:
        #        if mf.trainIdx == mb.queryIdx and mf.queryIdx == mb.trainIdx:
                    #if mf.trainIdx == mb.queryIdx:    # incorrect test - need to check both ways
        #            matches.append(mf)
        #            matches_backward.remove(mb)    # unique matching only

        #for r in rejected:
        #    print r.trainIdx, r.queryIdx

    else:
        matches = list(matches_forward)
        
    matches = [m for m in matches
                             if _are_close(kps1[m.queryIdx],
                            kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS)]


    return matches

def do_knn_match(kps1,kps2,desc1,desc2,crosscheck,proximity=True):
    # Do matching with BFMatch.kNNmatch(); with or without crosscheck, with or without proximity test

    match_candidates_forward = BFMATCH.knnMatch(desc1,desc2,k=K_NEAREST)
    matches_forward = []

    if not crosscheck:

        if not proximity:
            for knnlist in match_candidates_forward:
                matches_forward.append(knnlist[0])      
            return matches_forward
        else:   # do proximity
            for knnlist in match_candidates_forward:
                m = knnlist[0]
                if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS):
                    matches_forward.append(m)
            return matches_forward

    else:    # do cross-check

        match_candidates_backward = BFMATCH.knnMatch(desc2,desc1,k=K_NEAREST)
        matches_backward = []

        if not proximity:
            for knnlist in match_candidates_forward:
                matches_forward.append(knnlist[0])
            for knnlist in match_candidates_backward:
                matches_backward.append(knnlist[0])
            matches = []
            for mf in matches_forward:
                for mb in matches_backward:
                    if mf.trainIdx == mb.queryIdx and mf.queryIdx == mb.trainIdx:
                    #if mf.trainIdx == mb.queryIdx:    # incorrect test - need to check both ways
                        matches.append(mf)
                        matches_backward.remove(mb)    # unique matching only
                        break
            return matches

        else:   # do proximity
            for knnlist in match_candidates_forward:
                m = knnlist[0]
                if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS):
                    matches_forward.append(m)
            for knnlist in match_candidates_backward:
                m = knnlist[0]
                if _are_close(kps2[m.queryIdx],kps1[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS):
                    matches_backward.append(m)

            matches = []
    
            for mf in matches_forward:
                for mb in matches_backward:
                    if mf.trainIdx == mb.queryIdx and mf.queryIdx == mb.trainIdx:
                    #if mf.trainIdx == mb.queryIdx:    # incorrect test - need to check both ways
                        matches.append(mf)
                        matches_backward.remove(mb)    # unique matching only
                        break

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

# test a: no cross-checks:
matches_match = do_non_knn_match(kps1,kps2,desc1,desc2,False,False)
matches_knn = do_knn_match(kps1,kps2,desc1,desc2,False)
print("No Xcheck: .match(): {0}, .kNNmatch(): {1}".format(len(matches_match),len(matches_knn)))

# test b: yes CV2 cross-check:
matches_match = do_non_knn_match(kps1,kps2,desc1,desc2,True,False)
matches_knn = do_knn_match(kps1,kps2,desc1,desc2,True)
print("CV2 Xcheck: .match(): {0}, .kNNmatch(): {1}".format(len(matches_match),len(matches_knn)))

# test c: our cross-check:
matches_match = do_non_knn_match(kps1,kps2,desc1,desc2,False,True)
matches_knn = do_knn_match(kps1,kps2,desc1,desc2,True)
print("Our Xcheck: .match(): {0}, .kNNmatch(): {1}".format(len(matches_match),len(matches_knn)))


print("Finished")
