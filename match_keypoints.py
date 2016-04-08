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

import matplotlib.pyplot as plt


# global parameters
MATCH_PROXIMITY_IN_PIXELS = 4              # empirical
KAZE_PARAMETER = 0.0003                 	# empirical
#KAZE_PARAMETER = 0.001
FLANN_KDTREE_INDEX = 0                  	# definition
FLANN_TREE_NUMBER = 5                       # empirical
FLANN_SEARCH_DEPTH = 50                 	# empirical
IMAGE_WIDTH = 400                           # expected image width

def _gaussian(x,a,x0,sigma):
    # Defines a gaussian function for curve fitting
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

def _poisson(x,a,y0,lam):
    # Defines a poisson distribution for curve fitting
    return y0 + a * np.exp(-lam) * lam**x / factorial(x)

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
    lam_guess = np.argmax(offsets)
    a_guess = offsets[lam_guess]
    y0_guess = offsets[-1]
    print("Poisson guesses: a = {0}, lam = {1}".format(a_guess,lam_guess))
    popt,pcov = curve_fit(_poisson,xdat,offsets,[a_guess,lam_guess,y0_guess])
    print("Poisson fit: {0}".format(popt))
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
    lam_guess = np.argmax(offsets)
    a_guess = offsets[lam_guess]
    y0_guess = offsets[-1]
    print("Poisson guesses: a = {0}, lam = {1}".format(a_guess,lam_guess))
    popt,pcov = curve_fit(_poisson,xdat,offsets,[a_guess,lam_guess,y0_guess])
    print("Poisson fit: {0}".format(popt))

    return popt   # should eventually test inverse CDF or something


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
#SIFT = cv2.xfeatures2d.SIFT_create()        # use default features
#BFMATCH = cv2.BFMatcher()
KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
index_parameters = dict(algorithm = FLANN_KDTREE_INDEX, trees = FLANN_TREE_NUMBER)
search_parameters = dict(checks=FLANN_SEARCH_DEPTH)
FLANN = cv2.FlannBasedMatcher(index_parameters,search_parameters)
#AKAZE = cv2.AKAZE_create(threshold= KAZE_PARAMETER)

# find keypints and descriptors
#kps1,desc1 = SIFT.detectAndCompute(im1,None)
#kps2,desc2 = SIFT.detectAndCompute(im2,None)
kps1,desc1 = KAZE.detectAndCompute(im1,None)
kps2,desc2 = KAZE.detectAndCompute(im2,None)
#kps1,desc1 = AKAZE.detectAndCompute(im1,None)
#kps2,desc2 = AKAZE.detectAndCompute(im2,None)
print 'Found {0} kps in im1'.format(len(kps1))
print 'Found {0} kps in im2'.format(len(kps2))

# find 2-way matches
match_candidates = FLANN.match(desc1,desc2)
#match_candidates = BFMATCH.match(desc1,desc2)
print 'Found {0} match candidates...'.format(len(match_candidates))

# do proximity test
match_offsets = _calculate_offsets_between_matches(kps1,kps2,match_candidates)

# show match offset histogram and poisson match
_show_offset_histogram_and_poisson(match_offsets,im_size=IMAGE_WIDTH)


proximity_in_pixels = _calculate_proximity_threshold(match_offsets)
offsets_histogram = _make_offsets_histogram(match_offsets,MATCH_PROXIMITY_IN_PIXELS)

#MATCH_PROXIMITY_IN_PIXELS = proximity_in_pixels
matches = [
	m for m in match_candidates if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],MATCH_PROXIMITY_IN_PIXELS)
]
print '...of which {0} are within the proximity limit of {1} pixels.'.format(len(matches),MATCH_PROXIMITY_IN_PIXELS)
kps1_matched = [kps1[m.queryIdx] for m in matches]
kps2_matched = [kps2[m.trainIdx] for m in matches]
kps1_unmatched = set(kps1) - set(kps1_matched)
kps2_unmatched = set(kps2) - set(kps2_matched)

# calculate average match rate for each image
N_kps1 = len(kps1)
N_kps2 = len(kps2)
N_matches = len(matches)
match_rate_1 = N_matches/float(N_kps1)           #  average match rate for im1
match_rate_2 = N_matches/float(N_kps2)           #  average match rate for im2

# OUTPUT

# prepare output image
im1_out = im1_color.copy()
im2_out = im2_color.copy()
cv2.drawKeypoints(im1_color,kps1_matched,im1_out,color=match_color,flags=0)
cv2.drawKeypoints(im2_color,kps2_matched,im2_out,color=match_color,flags=0)
#cv2.drawKeypoints(im1_out,kps1_unmatched,im1_out,color=non_match_color,flags=0)
#cv2.drawKeypoints(im2_out,kps2_unmatched,im2_out,color=non_match_color,flags=0)

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
label_string = "KAZE (0.0003); proximity test @ {0} pixels; FLANN".format(MATCH_PROXIMITY_IN_PIXELS)
label_origin = (20,80)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)

# join label to bottom of image pair and save
im_A = np.concatenate((im1_out,im2_out),axis=1)
im_B = np.concatenate((im_A,text_box),axis=0)
im_C = np.concatenate((im_B,hist_plots),axis=0)
cv2.imwrite(args.save_image_filename,im_C)
