""" test_binomial_batch.py
Test whether keypoints are binomially distributed throughout an image.
:params:
: image_directory: Directory where images are saved (assumes PNG format)
: log_file: Filename for log
"""

import argparse
import numpy as np
import cv2
import scipy.stats
import os

# global parameters
NUMBER_OF_PATCHES_WIDE = 20
NUMBER_OF_PATCHES_HIGH = 16
KAZE_PARAMETER = 0.0003                 	# empirical
MIN_KP_COUNT = 1

def _count_keypoints_in_each_neighborhood(kps,im):
    # count keypoints in each patch defined by NUMBER_OF_PATCHES_{WIDE/HIGH}
    kp_count = np.zeros([NUMBER_OF_PATCHES_WIDE,NUMBER_OF_PATCHES_HIGH])
    for kp in kps:
        kp_count[kp.pt[1]/PATCH_WIDTH,kp.pt[0]/PATCH_HEIGHT] += 1

    return kp_count.flatten()

def _calculate_binom_cdf(kp_count):
    # calculate the CDF of a binomial distribution with the properties of kp_count
    max_count = kp_count.max()
    num_keypoints = kp_count.sum()
    prob = np.mean(kp_count) / float(num_keypoints)
    count_vals = np.arange(max_count)
    cdf_vals = scipy.stats.binom.cdf(count_vals,num_keypoints,prob)

    return count_vals, cdf_vals

def _calculate_ks_statistic(kp_cdf,binom_cdf):
    # calculate Kolmogorov-Smirnov statistic
    ks_vals = abs(kp_cdf - binom_cdf)
    return ks_vals.max()

# main

# read args
parser = argparse.ArgumentParser(description='Test whether keypoints are binomially distributed.')
parser.add_argument('image_directory', type=str, help='Path to directory where images are saved.')
parser.add_argument('log_file', type=str, help='Filename to log results.')
args = parser.parse_args()

# open log file
f_log = open(args.log_file,'a')

# instantiate KAZE object
KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)

# process each image in image_directory
for im_filename in os.listdir(args.image_directory):
    # try to load image file [note grayscale: 0; color: 1]
    if '.png' not in im_filename:
        continue
    try:
        im_raw = cv2.imread(os.path.join(args.image_directory,im_filename),0)
    except:
        print("Could not load file {0}, skipping.".format(im_filename))
        continue

    IMAGE_WIDTH, IMAGE_HEIGHT = im_raw.shape[0], im_raw.shape[1]
    PATCH_WIDTH = IMAGE_WIDTH / NUMBER_OF_PATCHES_WIDE
    PATCH_HEIGHT = IMAGE_HEIGHT / NUMBER_OF_PATCHES_HIGH

    # discard edges of image that are outside patch grid
    trimmed_width = PATCH_WIDTH * NUMBER_OF_PATCHES_WIDE
    trimmed_height = PATCH_HEIGHT * NUMBER_OF_PATCHES_HIGH
    im = np.zeros((trimmed_width,trimmed_height),dtype='uint8')
    im[:,:] = im_raw[0:trimmed_width,0:trimmed_height]

    # detect keypoints
    kps = KAZE.detect(im,None)

    # count keypoints in each neighborhood/patch
    kp_count = _count_keypoints_in_each_neighborhood(kps,im)
    if kp_count.sum() <= MIN_KP_COUNT:
        print("Skipping {0}, too few keypoints.".format(im_filename))
        continue

    # calculate theoretical binomial CDF
    count_vals, binom_cdf = _calculate_binom_cdf(kp_count)

    # calculate CDF for keypoints
    bin_edges = np.arange(kp_count.max()+1)
    kp_hist, bin_edges = np.histogram(kp_count,bin_edges,density=True)
    kp_cdf = np.cumsum(kp_hist)

    # calculate K-S statistic
    ks_stat = _calculate_ks_statistic(kp_cdf,binom_cdf)

    # print output
    print("File: {0}, N_kps: {1}, K-S stat: {2}".format(im_filename,len(kps),ks_stat))
    log_string = "{0}, {1}, {2}\n".format(im_filename,len(kps), ks_stat)
    f_log.write(log_string)

f_log.close()
print("Finished")
