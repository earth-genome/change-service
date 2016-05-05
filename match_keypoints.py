""" match_keypoints.py
Detect and match keypoints between two images.
Arguments:
- filename1: First image file
- filename2: Second image file
- filename3: Filename to save results
- date1: First image file date string
- date2: Second image file date string
"""

import argparse
import numpy as np
import cv2
import scipy.stats
from scipy.optimize import curve_fit
from scipy.misc import factorial
from scipy.special import gammaln
from scipy.stats import poisson
import random
import matplotlib.pyplot as plt

# global parameters
FEATURES = 'KAZE'
MATCH_PROXIMITY_IN_PIXELS = 4               # empirical
MATCH_NEIGHBORHOOD_IN_PIXELS = 40           # empirical
MIN_NEIGHBORHOOD_KEYPOINTS = 10
KAZE_PARAMETER = 0.0003                 	# empirical
#KAZE_PARAMETER = 0.001
K_NEAREST = 5                               # empirical
FLANN_KDTREE_INDEX = 0                  	# definition
FLANN_TREE_NUMBER = 5                       # empirical
FLANN_SEARCH_DEPTH = 50                 	# empirical
IMAGE_WIDTH = 512                           # expected image width

def _gaussian(x,a,x0,sigma):
    # Defines a gaussian function for curve fitting
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

def _poisson(x,a,y0,mu):
    # Defines a poisson distribution for curve fitting
    #return y0 + a * np.exp(-mu) * mu**x / factorial(x)
    #return y0 + a * np.exp(x * np.log(mu) - mu - gammaln(x+1))
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
    # Use this to identify cut-off for proximity test

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

def _show_offset_histogram_and_poisson(offsets,im_size=IMAGE_WIDTH):

    xdat = np.arange(len(offsets))
    mu_guess = np.argmax(offsets)
    a_guess = offsets[mu_guess]
    y0_guess = offsets[-1]
    print("Poisson guesses: a = {:3.1f}, mu = {:3.1f}, y0 = {:3.1f}".format(a_guess,mu_guess,y0_guess))
    popt,pcov = curve_fit(_poisson,xdat,offsets,[a_guess,mu_guess,y0_guess])
    print("Poisson fit: a = {:3.1f}, mu = {:3.1f}, y0 = {:3.1f}".format(popt[0],popt[1],popt[2]))
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(xdat, offsets,'ro')     
    ax1.plot(xdat, _poisson(xdat,popt[0],popt[1],popt[2]),'bx')     
    plt.show()

def _make_offsets_histogram(offsets,cutoff,im_size=IMAGE_WIDTH):
    # Make histogram plot of pixel offsets between matches
    # Use only OpenCV functions

    x_size = 1.0 * im_size   # extend across full images
    max_offset = int(0.25 * im_size)    
    x_scale_factor = x_size / max_offset
    y_size = 64.0   # display convenience only
    max_val = int(1.5 * np.amax(offsets)) 
    y_scale_factor = y_size / max_val
    plot_box = 255 * np.ones((y_size,x_size,3),np.uint8)
    for n,val in enumerate(offsets):
        xpos = int(x_scale_factor * n)
        ypos = int(y_size-1-y_scale_factor*val)             # remember inverted axes
        plot_box[ypos:y_size,xpos:xpos+3,:] = (0,0,0)       # color black
    x_cutoff = int(x_scale_factor*cutoff + 4)
    plot_box[:,x_cutoff,:] = (255,0,0)       # mark cutoff in blue
    mylabel = 'Match prox.; cutoff: {0} px'.format(cutoff)
    cv2.putText(plot_box,mylabel,(20,16),1,1.0,black_color)
    plot_box[0,:,:] = (0,0,0)       # top divider
    plot_box[:,-1,:] = (0,0,0)      # edge divider
    plot_box[:,0,:] = (0,0,0)       # edge divider
    return plot_box

def _plot_local_keypoint_histogram(kp_histogram_forward,kp_histogram_backward,im_size=IMAGE_WIDTH):
    # Returns a histogram plot of number of local keypoints
    # Illustrates how many keypoints are local to each unmatched keypoint

    x_size = 1.0 * im_size    # stretch across one image
    max_kps = 128.0           # arbitrary
    x_scale_factor = x_size / max_kps
    y_size = 64.0               # for convenience of display
    max_val = int(1.5 * np.amax(np.concatenate((kp_histogram_forward,kp_histogram_backward),axis=0)))
    y_scale_factor = y_size / max_val
    plot_box = 255 * np.ones((y_size,x_size,3),np.uint8)
    for n,val in enumerate(kp_histogram_forward):
        xpos = int(x_scale_factor * n)
        ypos = int(y_size-1-y_scale_factor*val)
        plot_box[ypos:y_size,xpos:xpos+1,:] = change_color_forward
    for n,val in enumerate(kp_histogram_backward):
        xpos = int(x_scale_factor * n)
        ypos = int(y_size-1-y_scale_factor*val)
        plot_box[ypos:y_size,xpos+2:xpos+3,:] = change_color_backward
    x_cutoff = int(x_scale_factor*MIN_NEIGHBORHOOD_KEYPOINTS+4)
    plot_box[:,x_cutoff,:] = (255,0,0) # mark MIN_NEIGHBORHOOD_KEYPOINTS
    mylabel = 'Local keypts.; MIN: {0}'.format(MIN_NEIGHBORHOOD_KEYPOINTS)
    cv2.putText(plot_box,mylabel,(20,16),1,1.0,black_color)
    plot_box[0,:,:] = (0,0,0)       # top divider
    plot_box[:,-1,:] = (0,0,0)      # edge divider
    return plot_box

def _calculate_proximity_threshold(offsets,num_sigma=3):
    # Calculates the optimum proximity threshold
    # Starts by fitting histogram of match offsets to a gaussian
    # The threshold is the center of the gaussian fit, plus num_sigma * sigma
    # Note the means we don't cut off matches that are at short distances; small effect
    # NOTE: Currently trying a possion fit instead of gaussian

    xdat = np.arange(len(offsets))
    
    # gaussian version
    #x0_guess = np.argmax(offsets)        
    #sigma_guess = 2     # empirical
    #popt,pcov = curve_fit(_gaussian,xdat,offsets,[a_guess,x0_guess,sigma_guess])
    #return int(popt[1] + num_sigma * popt[2])

    # poisson version
    mu_guess = np.argmax(offsets)
    a_guess = offsets[mu_guess]
    y0_guess = offsets[-1]
    popt,pcov = curve_fit(_poisson,xdat,offsets,[a_guess,mu_guess,y0_guess])
    if popt[1] > 0:
        max_distance = int(poisson.interval(0.95,popt[1])[-1])
    else:
        max_distance = MATCH_PROXIMITY_IN_PIXELS
        print("Fit did not converge.")
    #return int(4*popt[1])   # rough guess; should eventually test inverse CDF or something
    return max_distance

def _plot_matches(im1,im2,kps1,kps2,matches):

    im_out = np.concatenate((im1,im2),axis=1)
    cols = im1.shape[1]
    for m in matches:
        pt1 = (int(kps1[m.queryIdx].pt[0]),int(kps1[m.queryIdx].pt[1]))
        pt2 = (int(kps2[m.trainIdx].pt[0]+cols),int(kps2[m.trainIdx].pt[1]))
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.line(im_out,pt1,pt2,color=color)
    return im_out

# main

# set up color choices
black_color = (0,0,0)
match_color = (0,255,0)
non_match_color = (0,0,255)
marker_color = (255,0,0)

# read args
parser = argparse.ArgumentParser(description='Match keypoints between two images.')
parser.add_argument('image_1_filename', type=str, help='First image file name.')
parser.add_argument('image_2_filename', type=str, help='Second image file name.')
parser.add_argument('save_image_filename', type=str, help='Path to save output file.')
parser.add_argument('date1', type=str, help='First image file date.')
parser.add_argument('date2', type=str, help='Second image file date.')
args = parser.parse_args()

# load image files [note grayscale: 0; color: 1]
im1 = cv2.imread(args.image_1_filename,0)
im1_color = cv2.imread(args.image_1_filename,1)
im2 = cv2.imread(args.image_2_filename,0)
im2_color = cv2.imread(args.image_2_filename,1)

# instantiate global OpenCV objects
BFMATCH = cv2.BFMatcher(crossCheck = True)
if FEATURES == 'SIFT':
    SIFT = cv2.xfeatures2d.SIFT_create()
    kps1,desc1 = SIFT.detectAndCompute(im1,None)
    kps2,desc2 = SIFT.detectAndCompute(im2,None)
elif FEATURES == 'KAZE':
    KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
    kps1,desc1 = KAZE.detectAndCompute(im1,None)
    kps2,desc2 = KAZE.detectAndCompute(im2,None)
print 'Found {0} kps in im1'.format(len(kps1))
print 'Found {0} kps in im2'.format(len(kps2))

# find 2-way matches
match_candidates = BFMATCH.match(desc1,desc2)
print 'Found {0} match candidates...'.format(len(match_candidates))

# do proximity test
match_offsets = _calculate_offsets_between_matches(kps1,kps2,match_candidates)
proximity_in_pixels = _calculate_proximity_threshold(match_offsets)
offsets_histogram = _make_offsets_histogram(match_offsets,proximity_in_pixels)
matches = [
	m for m in match_candidates if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],proximity_in_pixels)
]
print '...of which {0} are within the proximity limit of {1} pixels.'.format(len(matches),proximity_in_pixels)
kps1_matched = [kps1[m.queryIdx] for m in matches]
kps2_matched = [kps2[m.trainIdx] for m in matches]

# calculate average match rate for each image
N_kps1 = len(kps1)
N_kps2 = len(kps2)
N_matches = len(matches)
average_match_rate = 2*N_matches/float(N_kps1+N_kps2)
print("Average match rate: {0}".format(average_match_rate))

#_show_offset_histogram_and_poisson(match_offsets)

"""
# REDO proximity test using KNN matching
match_candidates_k = BFMATCH.knnMatch(desc1,desc2,k=K_NEAREST)
print 'Found {0} match candidates with kNN...'.format(len(match_candidates_k))
matches_k = []
for knnlist in match_candidates_k:
    for m in knnlist:
        if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],proximity_in_pixels):
            matches_k.append(m)
            break
print '...of which {0} are within the proximity limit of {1} pixels.'.format(len(matches_k),proximity_in_pixels)
kps1_matched = [kps1[m.queryIdx] for m in matches_k]
kps2_matched = [kps2[m.trainIdx] for m in matches_k]
"""

# OUTPUT

# prepare output image
im1_out = im1_color.copy()
im2_out = im2_color.copy()
cv2.drawKeypoints(im1_color,kps1_matched,im1_out,color=match_color,flags=0)
cv2.drawKeypoints(im2_color,kps2_matched,im2_out,color=match_color,flags=0)

im_combined = np.concatenate((im1_out,im2_out),axis=1)
#im_combined = _plot_matches(im1_out,im2_out,kps1,kps2,matches)
#im_combined = cv2.drawMatches(im1_out,kps1,im2_out,kps2,matches)

# calculate histogram and plot
#local_keypoint_histogram = _plot_local_keypoint_histogram(kps2_local_histogram,kps1_local_histogram)
local_keypoint_histogram = offsets_histogram 
hist_plots = np.concatenate((offsets_histogram,local_keypoint_histogram),axis=1)

# make label box
text_box_height = 10 + 20 * 4
text_box = 255 * np.ones((text_box_height,2*IMAGE_WIDTH,3),np.uint8)

label_string = args.date1
label_origin = (20,20)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
label_string = args.date2
label_origin = (20+IMAGE_WIDTH,20)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
label_string = "Keypoints: {0}".format(len(kps1))
label_origin = (20,40)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
label_string = "Keypoints: {0}".format(len(kps2))
label_origin = (20+IMAGE_WIDTH,40)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
label_string = "Matched keypoints: {0}".format(len(kps2_matched))
label_origin = (20,60)
cv2.putText(text_box,label_string,label_origin,1,1.0,match_color)
label_string = "{0} ({1}); proximity test @ {2} pixels; BFMatch".format(FEATURES, KAZE_PARAMETER, proximity_in_pixels)
#label_string = "SIFT, BFMatch (cross-check), proximity test: {0}".format(proximity_in_pixels)
label_origin = (20,80)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)

# join label to bottom of image pair and save
#im_A = np.concatenate((im1_out,im2_out),axis=1)
im_A = im_combined
im_B = np.concatenate((im_A,text_box),axis=0)
im_C = np.concatenate((im_B,hist_plots),axis=0)
cv2.imwrite(args.save_image_filename,im_C)
