""" test_binomial.py
Test whether keypoints are binomially distributed throughout an image.
:params:
: image_directory: Directory where images are saved (assumes PNG format)
: log_file: Filename for log
"""

import argparse
import numpy as np
import cv2
import scipy.stats
#import matplotlib.pyplot as plt
import os

# global parameters
NUMBER_OF_PATCHES_WIDE = 20
NUMBER_OF_PATCHES_HIGH = 20
KAZE_PARAMETER = 0.0003                 	# empirical
MIN_KP_COUNT = 500
keypoint_color = (255,0,0)

def _count_keypoints_in_each_neighborhood(kps,im):
    # count keypoints in each patch defined by NUMBER_OF_PATCHES_{WIDE/HIGH}
    kp_count = np.zeros([NUMBER_OF_PATCHES_WIDE,NUMBER_OF_PATCHES_HIGH])
    for kp in kps:
        kp_count[kp.pt[0]/PATCH_WIDTH,kp.pt[1]/PATCH_HEIGHT] += 1

    return kp_count.flatten()

def _calculate_keypoint_count_pmf(kp_count):
    # calculate properties of kp_count distribution
    max_count = kp_count.max()
    num_keypoints = kp_count.sum()
    prob = np.mean(kp_count) / float(num_keypoints)

    # calculate expected binomial pmf
    count_vals = np.arange(max_count)
    pmf_vals = NUMBER_OF_PATCHES_HIGH * NUMBER_OF_PATCHES_WIDE * scipy.stats.binom.pmf(count_vals,num_keypoints,prob)

    # calculate chi-squared
    kp_count_histogram = np.histogram(kp_count,max_count+1)[0]
    chi2,p = scipy.stats.chisquare(kp_count_histogram,pmf_vals)

    return pmf_vals,chi2,p

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

def _prepare_image(im,kps):
    # draw grid lines and keypoints on image
    im_out = im.copy()
    for x in np.arange(NUMBER_OF_PATCHES_WIDE):
        im[:,x*PATCH_WIDTH] = 0
    for y in np.arange(NUMBER_OF_PATCHES_HIGH):
        im[y*PATCH_HEIGHT,:] = 0
    cv2.drawKeypoints(im,kps,im_out,color=keypoint_color,flags=0)
    return im_out

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
        print("Could not load file {0}, skipping.".format(image))
        continue

    IMAGE_HEIGHT, IMAGE_WIDTH = im_raw.shape[0], im_raw.shape[1]
    #print("Image width: {0}; image height: {1}".format(IMAGE_WIDTH,IMAGE_HEIGHT))
    PATCH_WIDTH = IMAGE_WIDTH / NUMBER_OF_PATCHES_WIDE
    PATCH_HEIGHT = IMAGE_HEIGHT / NUMBER_OF_PATCHES_HIGH
    #print("Patch size: {0} wide, {1} high".format(PATCH_WIDTH,PATCH_HEIGHT))

    # discard edges of image that are outside patch grid
    trimmed_width = PATCH_WIDTH * NUMBER_OF_PATCHES_WIDE
    trimmed_height = PATCH_HEIGHT * NUMBER_OF_PATCHES_HIGH
    im = np.zeros((trimmed_height,trimmed_width),dtype='uint8')
    im[:,:] = im_raw[0:trimmed_height,0:trimmed_width]
    #print("Trimmed image to width: {0}, height: {1}".format(trimmed_width,trimmed_height))

    # detect keypoints
    kps = KAZE.detect(im,None)
    #print 'Found {0} kps in image'.format(len(kps))

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
    #print("K-S statistic: {0}".format(ks_stat))

    # print output
    print("File: {0}, N_kps: {1}, K-S stat: {2}".format(im_filename,len(kps),ks_stat))
    log_string = "{0}, {1}, {2}\n".format(im_filename,len(kps), ks_stat)
    f_log.write(log_string)

f_log.close()
print("Finished")

# build figure 
#fig = plt.figure(figsize=(16,6))

# plot keypoints-per-patch histogram and expected pmf
#ax1 = fig.add_subplot(1,2,1)
#ax1.plot(count_vals+0.5, kp_cdf,'bx')
#ax1.plot(count_vals+0.5, binom_cdf,'ro')     # offset x vals by 0.5 to center point on bar
#ax1.set_xlabel("Keypoints in patch")
#ax1.set_ylabel("Cumulative fraction of patches")

# add image with keypoints
#im_drawn = _prepare_image(im,kps)
#ax2 = fig.add_subplot(1,2,2)
#ax2.imshow(im_drawn)

# display    
#plt.show()