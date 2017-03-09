""" test_binomial.py
Test whether keypoints are binomially distributed throughout an image.
:params:
: image_file: Filename of image
: save_file: Filename for saved K-S graph
: save_image_file: Filename for saved gridded image with keypoints
"""

import argparse
import numpy as np
import cv2
import scipy.stats
import matplotlib.pyplot as plt

# global parameters
NUMBER_OF_PATCHES_WIDE = 20
NUMBER_OF_PATCHES_HIGH = 16
KAZE_PARAMETER = 0.0003                 	# empirical
keypoint_color = (127,0,0)
black_color = (0,0,0)

def _count_keypoints_in_each_neighborhood(kps,im):
    # count keypoints in each patch defined by NUMBER_OF_PATCHES_{WIDE/HIGH}
    kp_count = np.zeros([NUMBER_OF_PATCHES_WIDE,NUMBER_OF_PATCHES_HIGH])
    for kp in kps:
        kp_count[int(kp.pt[1]/PATCH_WIDTH),int(kp.pt[0]/PATCH_HEIGHT)] += 1

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
        im_out[x*PATCH_WIDTH,:] = black_color
    for y in np.arange(NUMBER_OF_PATCHES_HIGH):
        im_out[:,y*PATCH_HEIGHT] = black_color
    cv2.drawKeypoints(im_out,kps,im_out,color=keypoint_color,flags=0)
    return im_out

# main

# read args
parser = argparse.ArgumentParser(description='Test whether keypoints are binomially distributed.')
parser.add_argument('image_file', type=str, help='Image filename.')
parser.add_argument('save_file', type=str, help='Filename to save result.')
parser.add_argument('save_image_file', type=str, help='Filename to save image result.')
args = parser.parse_args()

# instantiate KAZE object
KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)

# try to load image file [note grayscale: 0; color: 1; with alpha channel: cv2.IMREAD_UNCHANGED]
im_raw = cv2.imread(args.image_file,0)
im_color = cv2.imread(args.image_file,1)
im_unchanged = cv2.imread(args.image_file,cv2.IMREAD_UNCHANGED)
IMAGE_WIDTH, IMAGE_HEIGHT = im_raw.shape[0], im_raw.shape[1]
print("Image width: {0}; image height: {1}".format(IMAGE_WIDTH,IMAGE_HEIGHT))
PATCH_WIDTH = IMAGE_WIDTH / NUMBER_OF_PATCHES_WIDE
PATCH_HEIGHT = IMAGE_HEIGHT / NUMBER_OF_PATCHES_HIGH
print("Patch size: {0} wide, {1} high".format(PATCH_WIDTH,PATCH_HEIGHT))

# discard edges of image that are outside patch grid
trimmed_width = PATCH_WIDTH * NUMBER_OF_PATCHES_WIDE
trimmed_height = PATCH_HEIGHT * NUMBER_OF_PATCHES_HIGH
print("Trimmed image: {0} wide, {1} high".format(trimmed_width,trimmed_height))
im = np.zeros((trimmed_width,trimmed_height),dtype='uint8')
im[:,:] = im_raw[0:trimmed_width,0:trimmed_height]

# detect keypoints
kps = KAZE.detect(im,None)
print 'Found {0} kps in image'.format(len(kps))

# count keypoints in each neighborhood/patch
kp_count = _count_keypoints_in_each_neighborhood(kps,im)

# calculate theoretical binomial CDF
count_vals, binom_cdf = _calculate_binom_cdf(kp_count)

# calculate CDF for keypoints
bin_edges = np.arange(kp_count.max()+1)
kp_hist, bin_edges = np.histogram(kp_count,bin_edges,density=True)
kp_cdf = np.cumsum(kp_hist)

# calculate K-S statistic
ks_stat = _calculate_ks_statistic(kp_cdf,binom_cdf)

# print output
print("File: {0}, N_kps: {1}, K-S stat: {2}".format(args.image_file,len(kps),ks_stat))

# build figure 
fig = plt.figure(figsize=(5,3.5))

# plot keypoints-per-patch histogram and expected pmf
ax1 = fig.add_subplot(1,1,1)
ax1.plot(count_vals+0.5, kp_cdf,'bs')
ax1.plot(count_vals+0.5, binom_cdf,'ro')     # offset x vals by 0.5 to center point on bar
ax1.set_xlabel("Keypoints in patch")
ax1.set_ylabel("Cumulative fraction of patches")

# save
fig.savefig(args.save_file, bbox_inches='tight')

# display    
#plt.show()

# save image with keypoints and gridlines
cv2.imwrite(args.save_image_file,_prepare_image(im_unchanged,kps))

"""
# add image with keypoints
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
im_drawn = _prepare_image(im_color,kps)
ax2.imshow(im_drawn,alpha=0.8)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
fig2.savefig(args.save_image_file, bbox_inches='tight', transparent=True, facecolor = fig.get_facecolor())
plt.show()
"""