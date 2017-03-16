""" match_keypoints_batch_new.py
Batch detect and match keypoints between two images.
:params:
: image_directory: Directory where images are saved (assumes PNG format); from only 2010 and 2012
: log_file: Filename for log
"""

import argparse
import numpy as np
import cv2
import os

# global parameters
FEATURES = ("SIFT", "KAZE")[1]              # choose feature type
KAZE_PARAMETER = 0.0003                     # empirical
MATCH_PROXIMITY_IN_PIXELS = 4               # empirical
K_NEAREST = 5                               # empirical
IMAGE_WIDTH = 512


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

def do_knn_match(kps1,kps2,desc1,desc2):
    # Do matching with kNN search; use fixed proximity test

    match_candidates = BFMATCH.knnMatch(desc1,desc2,k=K_NEAREST)
    matches_forward = []
    n_not_first_match_forward = 0
    for knnlist in match_candidates:
        for n,m in enumerate(knnlist):
            if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS):
                matches_forward.append(m)
                if n > 0:
                    n_not_first_match_forward += 1
                break   # only pick first match that passes proximity test (closest in descriptor space)

    n_forward = len(matches_forward)

    match_candidates = BFMATCH.knnMatch(desc2,desc1,k=K_NEAREST)
    matches_backward = []
    n_not_first_match_backward = 0
    for knnlist in match_candidates:
        for n,m in enumerate(knnlist):
            if _are_close(kps1[m.trainIdx],kps2[m.queryIdx],MATCH_PROXIMITY_IN_PIXELS):
                matches_backward.append(m)
                if n > 0:
                    n_not_first_match_backward += 1
                break   # only pick first match that passes proximity test (closest in descriptor space)

    n_backward = len(matches_backward)

    # cross-check forward and backward match lists
    matches = []
    for mf in matches_forward:
        for mb in matches_backward:
            if mf.trainIdx == mb.queryIdx and mf.queryIdx == mb.trainIdx:
            #if mf.trainIdx == mb.queryIdx:    # incorrect test - need to check both ways
                matches.append(mf)
                #matches_backward.remove(mb)    # unique matching only - may add later to speed up search
                break

    # test BFMatch with crosscheck
    """
    match_candidates_cc = BFMATCH_CC.match(desc1,desc2)
    matches_cc = [
        m for m in match_candidates_cc if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS)
    ]

    print("Num kNN matches: {0}; Num cc matches: {1}".format(len(matches),len(matches_cc)))
    """

    return matches, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward


# read args
parser = argparse.ArgumentParser(description='Match keypoints between two images.')
parser.add_argument('image_directory', type=str, help='Directory in which images are saved.')
parser.add_argument('log_file', type=str, help='Log file.')
args = parser.parse_args()

# open log file
f_log = open(args.log_file,'a')
log_string = "Filename, N_kps1, N_kps2, Total_kps2, N_matches, average_match_rate, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward"
f_log.write(log_string+'\n')

# instantiate global OpenCV objects
if FEATURES == "SIFT":
    SIFT = cv2.xfeatures2d.SIFT_create()
    feature_detector = SIFT       
elif FEATURES == "KAZE":
    KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
    feature_detector = KAZE

BFMATCH = cv2.BFMatcher()       # do not use cross-check if using k-nearest-neighbors test
BFMATCH_CC = cv2.BFMatcher(crossCheck=True)

# process each image in image_directory
for im_filename in os.listdir(args.image_directory):

    # only look at image files; trigger comparison starting with 2010 image only
    if '.png' not in im_filename or '2010' not in im_filename:
        continue    
    try:
        im1 = cv2.imread(os.path.join(args.image_directory,im_filename),0)
        im_filename_2012 = im_filename.replace('2010','2012')
        im2 = cv2.imread(os.path.join(args.image_directory,im_filename_2012),0)
    except:
        print("Could not load file {0}, skipping.".format(im_filename))
        continue

    # detect keypoints and descriptors
    kps1,kps2,desc1,desc2 = do_keypoint_detection(im1,im2,feature_detector)

    # find matches
    matches, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward = do_knn_match(kps1,kps2,desc1,desc2)

    # calculate average match rate 
    N_kps1 = len(kps1)
    N_kps2 = len(kps2)
    N_matches = len(matches)
    average_match_rate = 2*N_matches/float(N_kps1+N_kps2)       # this is the definition we use in the paper

    # print output
    log_string = "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}".format(im_filename,
        N_kps1,N_kps2,N_kps1+N_kps2,N_matches,average_match_rate,
        n_forward,n_backward,n_not_first_match_forward,n_not_first_match_backward)
    print log_string
    f_log.write(log_string+'\n')

f_log.close()

# display averages
match_rates = np.genfromtxt(args.log_file,delimiter=',',usecols=5,skip_header=1)
print("Average match rate: {0}".format(np.mean(match_rates)))

Num_kps1 = np.genfromtxt(args.log_file,delimiter=',',usecols=1,skip_header=1)
Num_kps2 = np.genfromtxt(args.log_file,delimiter=',',usecols=2,skip_header=1)
Num_forward_matches = np.genfromtxt(args.log_file,delimiter=',',usecols=6,skip_header=1)
Num_backward_matches = np.genfromtxt(args.log_file,delimiter=',',usecols=7,skip_header=1)
prox_test_pass_rate_forward = np.mean(np.divide(Num_forward_matches,Num_kps1))
prox_test_pass_rate_backward = np.mean(np.divide(Num_backward_matches,Num_kps2))
print("Average rate of passing proximity test (k={0}), forward: {1}, backward: {2}.".format(K_NEAREST,
    prox_test_pass_rate_forward,prox_test_pass_rate_backward))

Num_matches = np.genfromtxt(args.log_file,delimiter=',',usecols=4,skip_header=1)
crosscheck_vals = 2 * np.divide(Num_matches,(Num_forward_matches + Num_backward_matches))
print("Average rate of cross-check in kNN match: {0}".format(np.mean(crosscheck_vals)))

Num_not_first_match_forward = np.genfromtxt(args.log_file,delimiter=',',usecols=8,skip_header=1)
Num_not_first_match_backward = np.genfromtxt(args.log_file,delimiter=',',usecols=9,skip_header=1)
fraction_not_first_forward = np.divide(Num_not_first_match_forward,Num_forward_matches)
fraction_not_first_backward = np.divide(Num_not_first_match_backward,Num_backward_matches)
print("Average rate that not-first match was chosen: {0} (forward), {1} (backward)".format(np.mean(fraction_not_first_forward),
    np.mean(fraction_not_first_backward)))

print("Finished")
