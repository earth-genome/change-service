""" test_binomial.py
Test whether keypoints are binomially distributed throughout an image.
Arguments:
- image_filename: Image file
- save_image_filename: Filename to save results
"""

import argparse
import numpy as np
import cv2
from scipy.stats import binom
from scipy.stats import kstest
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# global parameters
NUMBER_OF_PATCHES_WIDE = 16
NUMBER_OF_PATCHES_HIGH = 24
KAZE_PARAMETER = 0.0003                 	# empirical
keypoint_color = (255,0,0)

def _count_keypoints_in_each_neighborhood(kps,im):
    # count keypoints in each patch defined by NUMBER_OF_PATCHES_{WIDE/HIGH}

    # build list of paths enclosing each equal-sized rectangular patch in image
    #height,width = im.shape
    dw = IMAGE_WIDTH / NUMBER_OF_PATCHES_WIDE
    dh = IMAGE_HEIGHT / NUMBER_OF_PATCHES_HIGH
    print("Patch size: {0} wide, {1} high".format(dw,dh))
    path_list = []
    for x in np.arange(NUMBER_OF_PATCHES_WIDE):
        for y in np.arange(NUMBER_OF_PATCHES_HIGH):
            verts = [ (x*dw,y*dh), (x*dw,(y+1)*dh), ((x+1)*dw,(y+1)*dh),((x+1)*dw,y*dh)]
            path_list.append(Path(verts))

    # test each keypoint to see which patch it's in (not efficient!)
    kp_count = np.zeros(len(path_list))
    for kp in kps:
        for n,patch in enumerate(path_list):
            if patch.contains_point(kp.pt):
                kp_count[n] += 1
    return kp_count, path_list

def _calculate_keypoint_count_pmf(kp_count):

    # calculate properties of kp_count distribution
    max_count = kp_count.max()
    num_keypoints = kp_count.sum()
    prob = np.mean(kp_count) / float(num_keypoints)
    print("Keypoint distribution in patches: Max: {0}, number: {1}, prob: {2}".format(max_count,num_keypoints,prob))

    # calculate expected binomial pmf
    count_vals = np.arange(max_count+1)
    pmf_vals = NUMBER_OF_PATCHES_HIGH * NUMBER_OF_PATCHES_WIDE * binom.pmf(count_vals,num_keypoints,prob)
    return pmf_vals,count_vals,num_keypoints,prob

# main

# read args
parser = argparse.ArgumentParser(description='Test whether keypoints are binomially distributed.')
parser.add_argument('image_filename', type=str, help='Image file name.')
parser.add_argument('save_image_filename', type=str, help='Path to save output file.')
args = parser.parse_args()

# load image file [note grayscale: 0; color: 1]
im = cv2.imread(args.image_filename,0)
im_color = cv2.imread(args.image_filename,1)
IMAGE_HEIGHT = im.shape[0]
IMAGE_WIDTH = im.shape[1]
print("Image width: {0}; image height: {1}".format(IMAGE_WIDTH,IMAGE_HEIGHT))

# instantiate KAZE object and find keypoints
KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
kps = KAZE.detect(im,None)
print 'Found {0} kps in image'.format(len(kps))

# count keypoints in each neighborhood/patch
kp_count,path_list = _count_keypoints_in_each_neighborhood(kps,im)

# calculate theoretical binomial pmf
pmf_vals,count_vals,num_keypoints,prob = _calculate_keypoint_count_pmf(kp_count)

# build figure 
fig = plt.figure(figsize=(16,6))

# plot keypoints-per-patch histogram and expected pmf
ax1 = fig.add_subplot(1,2,1)
max_count = kp_count.max()
count_vals = np.arange(max_count+1)
kp_count_histogram, bins, patch_silent_list = ax1.hist(kp_count, bins = max_count+1, range=(0,max_count+1), histtype = 'bar')
ax1.plot(count_vals+0.5, pmf_vals,'ro')     # offset x vals by 0.5 to center point on bar

# set axis labels and title
ax1.set_xlabel("Keypoints in patch")
ax1.set_ylabel("Number of patches")

# calculate Kolmogorov-Smirnov test value
#kp_count_histogram_normalized = kp_count_histogram / ( NUMBER_OF_PATCHES_HIGH * NUMBER_OF_PATCHES_WIDE)
#ks_val,p_val = kstest(kp_count_histogram_normalized,"binom.pdf",args=[count_vals,num_keypoints,prob])
#print("KS test: {0}".format(ks_val))

# add image with keypoints
im_out = im_color.copy()
cv2.drawKeypoints(im_color,kps,im_out,color=keypoint_color,flags=0)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(im_out)

# add patch outlines to image
for apath in path_list:
    patch = patches.PathPatch(apath, facecolor='None', edgecolor='black', lw=1)
    ax2.add_patch(patch)
    ax2.set_xlim(0,IMAGE_WIDTH)
    ax2.set_ylim(0,IMAGE_HEIGHT)
    ax2.xaxis.set_ticks([])
    ax2.yaxis.set_ticks([])

# display    
plt.show()
    

