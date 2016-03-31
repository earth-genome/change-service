""" test_binomial.py
Test whether keypoints are binomially distributed throughout an image.
Arguments:
- image_filename: Image file
- save_image_filename: Filename to save results
"""

import argparse
import numpy as np
import cv2
import scipy.stats
import matplotlib.pyplot as plt

# global parameters
NUMBER_OF_PATCHES_WIDE = 16
NUMBER_OF_PATCHES_HIGH = 24
KAZE_PARAMETER = 0.0003                 	# empirical
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
    count_vals = np.arange(max_count+1)
    pmf_vals = NUMBER_OF_PATCHES_HIGH * NUMBER_OF_PATCHES_WIDE * scipy.stats.binom.pmf(count_vals,num_keypoints,prob)

    # calculate chi-squared
    kp_count_histogram = np.histogram(kp_count,max_count+1)[0]
    chi2,p = scipy.stats.chisquare(kp_count_histogram,pmf_vals)

    return pmf_vals,chi2,p

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
parser.add_argument('image_filename', type=str, help='Image file name.')
parser.add_argument('save_image_filename', type=str, help='Path to save output file.')
args = parser.parse_args()

# load image file [note grayscale: 0; color: 1]
im = cv2.imread(args.image_filename,1)
IMAGE_HEIGHT = im.shape[0]
IMAGE_WIDTH = im.shape[1]
print("Image width: {0}; image height: {1}".format(IMAGE_WIDTH,IMAGE_HEIGHT))
PATCH_WIDTH = IMAGE_WIDTH / NUMBER_OF_PATCHES_WIDE
PATCH_HEIGHT = IMAGE_HEIGHT / NUMBER_OF_PATCHES_HIGH
print("Patch size: {0} wide, {1} high".format(PATCH_WIDTH,PATCH_HEIGHT))

# instantiate KAZE object and find keypoints
KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
kps = KAZE.detect(im,None)
print 'Found {0} kps in image'.format(len(kps))

# count keypoints in each neighborhood/patch
kp_count = _count_keypoints_in_each_neighborhood(kps,im)

# calculate theoretical binomial pmf
pmf_vals,chi2,p_val = _calculate_keypoint_count_pmf(kp_count)

print("Binomial fit: chi2 = {0}, p val = {1}".format(chi2,p_val))

# build figure 
fig = plt.figure(figsize=(16,6))

# plot keypoints-per-patch histogram and expected pmf
ax1 = fig.add_subplot(1,2,1)
max_count = kp_count.max()
count_vals = np.arange(max_count+1)
kp_count_histogram, bins, patch_silent_list = ax1.hist(kp_count,bins=max_count+1,range=(0,max_count+1),histtype ='bar')
ax1.plot(count_vals+0.5, pmf_vals,'ro')     # offset x vals by 0.5 to center point on bar
ax1.set_xlabel("Keypoints in patch")
ax1.set_ylabel("Number of patches")

# add image with keypoints
im_drawn = _prepare_image(im,kps)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(im_drawn)

# display    
plt.show()
