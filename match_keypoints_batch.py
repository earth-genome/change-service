""" match_keypoints_batch.py
Batch detect and match keypoints between two images.
:params:
: image_directory: Directory where images are saved (assumes PNG format); from only 2010 and 2012
: log_file: Filename for log
"""

import argparse
import numpy as np
import cv2
import os
import scipy.stats
from scipy.optimize import curve_fit
from scipy.misc import factorial
from scipy.stats import poisson

# global parameters
FEATURES = ("SIFT", "KAZE")[1]              # choose feature type
DO_DYNAMIC_PROXIMITY_TEST = False           # select fixed or dynamic proximity test
DO_KNN_MATCH = True                         # use kNN matching (DO_DYNAMIC_PROXIMITY_TEST must be False)
KAZE_PARAMETER = 0.0003                     # empirical
MATCH_PROXIMITY_IN_PIXELS = 4               # empirical
K_NEAREST = 5                               # empirical
IMAGE_WIDTH = 512


def _poisson(x,a,y0,mu):
    # Defines a poisson distribution for curve fitting
    return y0 + a * poisson.pmf(x,mu)

def _are_close(kpa,kpb,distance):
	# Returns true if keypoints are separated by less than distance

    a_coords = np.array([kpa.pt[0], kpa.pt[1]])
    b_coords = np.array([kpb.pt[0], kpb.pt[1]])
    dist = np.linalg.norm(a_coords - b_coords)
    if dist > 0 and dist < distance:
        return True
    else:
        return False

def _calculate_offsets_between_matches(kps1,kps2,match_candidates,im_size=IMAGE_WIDTH):
    # Returns array with number of match pairs at each pixel separation

    max_offset = int(0.25 * im_size)  # offsets should never be larger than this
    offsets = np.zeros((max_offset,),dtype=np.int)
    for m in match_candidates:
        pa = kps1[m.queryIdx]
        pb = kps2[m.trainIdx]
        pa_coords = np.array([pa.pt[0], pa.pt[1]])
        pb_coords = np.array([pb.pt[0], pb.pt[1]])
        dist = int(np.linalg.norm(pa_coords - pb_coords))
        if dist < max_offset:
            offsets[dist] += 1      # ignore larger offsets
    return offsets

def _calculate_proximity_threshold(offsets):
    # Returns the optimum proximity threshold

    xvals = np.arange(len(offsets))
    mu_guess = np.argmax(offsets)
    a_guess = offsets[mu_guess]
    y0_guess = offsets[-1]
    popt,pcov = curve_fit(_poisson,xvals,offsets,[a_guess,mu_guess,y0_guess])
    if popt[1] > 0:
        max_distance = int(poisson.interval(0.99,popt[1])[-1])
        fit_converged = 1
    else:
        max_distance = MATCH_PROXIMITY_IN_PIXELS
        fit_converged = 0
    return max_distance, fit_converged

def do_keypoint_detection(im1,im2,feature_detector):
    # Returns keypoints and descriptors for selected feature type

    kps1,desc1 = feature_detector.detectAndCompute(im1,None)
    kps2,desc2 = feature_detector.detectAndCompute(im2,None)

    return kps1,kps2,desc1,desc2

def do_non_knn_match(kps1,kps2,desc1,desc2,MATCHER):
    # Do matching without kNN search
    # Use either fixed or dynamic proximity test

    match_candidates = MATCHER.match(desc1,desc2)
    if DO_DYNAMIC_PROXIMITY_TEST:
        match_offsets = _calculate_offsets_between_matches(kps1,kps2,match_candidates)
        proximity_in_pixels,fit_converged = _calculate_proximity_threshold(match_offsets)
    else:
        proximity_in_pixels = MATCH_PROXIMITY_IN_PIXELS
        fit_converged = -1
    matches = [
        m for m in match_candidates if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],proximity_in_pixels)
    ]

    return matches, proximity_in_pixels, fit_converged

def do_knn_match(kps1,kps2,desc1,desc2,MATCHER):
    # Do matching with kNN search
    # Use fixed proximity test

    match_candidates = MATCHER.knnMatch(desc1,desc2,k=K_NEAREST)
    matches_forward = []
    n_not_first_match_forward = 0
    for knnlist in match_candidates:
        for n,m in enumerate(knnlist):
            if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS):
                matches_forward.append(m)
                if n > 0:
                    n_not_first_match_forward += 1
                break

    n_forward = len(matches_forward)

    match_candidates = MATCHER.knnMatch(desc2,desc1,k=K_NEAREST)
    matches_backward = []
    n_not_first_match_backward = 0
    for knnlist in match_candidates:
        for n,m in enumerate(knnlist):
            if _are_close(kps1[m.trainIdx],kps2[m.queryIdx],MATCH_PROXIMITY_IN_PIXELS):
                matches_backward.append(m)
                if n > 0:
                    n_not_first_match_backward += 1
                break

    n_backward = len(matches_backward)

    # cross-check forward and backward match lists
    matches = []
    for mf in matches_forward:
        for mb in matches_backward:
            if mf.trainIdx == mb.queryIdx:
                matches.append(mf)
                matches_backward.remove(mb)    # unique matching only
                break

    return matches, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward


# read args
parser = argparse.ArgumentParser(description='Match keypoints between two images.')
parser.add_argument('image_directory', type=str, help='Directory in which images are saved.')
parser.add_argument('log_file', type=str, help='Log file.')
args = parser.parse_args()

# open log file
f_log = open(args.log_file,'a')
log_string = "Filename, N_kps1, N_kps2, average_match_rate, proximity_in_pixels, fit_converged, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward"
f_log.write(log_string+'\n')

# instantiate global OpenCV objects
if FEATURES == "SIFT":
    SIFT = cv2.xfeatures2d.SIFT_create()
    feature_detector = SIFT       
elif FEATURES == "KAZE":
    KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
    feature_detector = KAZE

if DO_KNN_MATCH:
    BFMATCH = cv2.BFMatcher()       # do not use cross-check if using k-nearest-neighbors test
else:
    BFMATCH = cv2.BFMatcher(crossCheck=True)    

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
    if DO_KNN_MATCH:       # using k-nearest-neighbor matching
        matches, n_forward, n_backward, n_not_first_match_forward, n_not_first_match_backward = do_knn_match(kps1,kps2,desc1,desc2,BFMATCH)
        proximity_in_pixels = MATCH_PROXIMITY_IN_PIXELS
        fit_converged = -1
    else:           # not using k-nearest-neighbor matching
        matches, proximity_in_pixels, fit_converged = do_non_knn_match(kps1,kps2,desc1,desc2,BFMATCH)
        n_forward = -1
        n_backward = -1
        n_not_first_match_forward = -1
        n_not_first_match_backward = -1

    # calculate average match rate 
    N_kps1 = len(kps1)
    N_kps2 = len(kps2)
    N_matches = len(matches)
    average_match_rate = 2*N_matches/float(N_kps1+N_kps2)

    # print output
    log_string = "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}".format(im_filename,N_kps1,N_kps2,average_match_rate,proximity_in_pixels,fit_converged,n_forward,n_backward,n_not_first_match_forward,n_not_first_match_backward)
    print log_string
    f_log.write(log_string+'\n')

f_log.close()

# display averages
match_rates = np.genfromtxt(args.log_file,delimiter=',',usecols=3,skip_header=1)
print("Average match rate: {0}".format(np.mean(match_rates)))

if DO_DYNAMIC_PROXIMITY_TEST:
    fit_converged_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=5,skip_header=1)
    print("Fit converged rate: {0}".format(np.mean(fit_converged_vals)))
    proximity_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=4,skip_header=1)
    print("Average proximity: {0} pixels".format(np.mean(proximity_vals)))

if DO_KNN_MATCH:
    N_kps1_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=1,skip_header=1)
    N_kps2_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=2,skip_header=1)
    N_matches_vals = 0.5 * np.multiply((N_kps1_vals + N_kps2_vals),match_rates)
    N_forward_matches_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=6,skip_header=1)
    N_backward_matches_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=7,skip_header=1)
    crosscheck_vals = 2 * np.divide(N_matches_vals,(N_forward_matches_vals + N_backward_matches_vals))
    print("Average cross-check rate in kNN match: {0}".format(np.mean(crosscheck_vals)))

    N_not_first_match_forward_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=8,skip_header=1)
    N_not_first_match_backward_vals = np.genfromtxt(args.log_file,delimiter=',',usecols=9,skip_header=1)
    fraction_not_first_forward = np.divide(N_not_first_match_forward_vals,N_forward_matches_vals)
    fraction_not_first_backward = np.divide(N_not_first_match_backward_vals,N_backward_matches_vals)
    print("Rate that not-first match was chosen: {0} (forward), {1} (backward)".format(np.mean(fraction_not_first_forward),np.mean(fraction_not_first_backward)))

print("Finished")
