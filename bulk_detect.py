""" Work towards streamlining detect_changes.py for bulk runs.

External function: detect_change()
    Arguments (current defaults are set in bulk_wrapper.py):
                  im1_file,
                  im2_file,
                    RelDir,   #path from pwd to image directory
                    output_dir,
                    **kwparams to include:
                    FEATURES,
                    KAZE_PARAMETER,
                    KNEAREST,
                    MATCH_PROXIMITY_IN_PIXELS,
                    CALCULATE_PROXIMITY_LIMIT, # overrides prev.
                    MATCH_NEIGHBORHOOD_IN_PIXELS,
                    MATCH_PROBABILITY_THRESHOLD

    Assumptions:
    -- im1_file and im2_file have the same shape.
    -- To get output filename and dates to format correctly, the image
    filenames should have format: basename-date.jpg,
    e.g. losangeles005-2010.jpg.

    Output: An image file with the two images, change points, and
    parameters displayed.

    Example: detect_change(losangeles005-2010.jpg,losangeles005-2012.jpg,
        **bulk_wrapper.default_change_params)

    
"""
import sys
import os
import numpy as np
import cv2
import scipy.stats
import scipy.signal
from scipy.optimize import curve_fit
from scipy.misc import factorial
from scipy.stats import poisson
import pdb

import bulk_wrapper

# set up color choices
black_color = (0,0,0)
change_color_forward = (0,0,255)
change_color_backward = (0,255,0)

# possible global parameters with defaults as of 4/5/16
#MATCH_PROXIMITY_IN_PIXELS = 4              # empirical
#MATCH_NEIGHBORHOOD_IN_PIXELS = 40       	# empirical
#MIN_NEIGHBORHOOD_KEYPOINTS = 10              # empirical
#MATCH_PROBABILITY_THRESHOLD = 1e-8      	# empirical
#KAZE_PARAMETER = 0.0003                 	# empirical
#FLANN_KDTREE_INDEX = 0                  	# definition
#FLANN_TREE_NUMBER = 5                       # empirical
#FLANN_SEARCH_DEPTH = 50                 	# empirical
#KNEAREST = 5  # For relaxed matching, proximity test the KNEAREST matches

def _gaussian(x,a,x0,sigma):
    # Defines a gaussian function for curve fitting
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

def _calculate_offsets_between_matches(kps1,kps2,match_candidates,im_size):
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

def _make_offsets_histogram(offsets,cutoff,im_size):
    # Make histogram plot of pixel offsets between matches
    # Use only OpenCV functions

    x_size = 1.0 * im_size   # extend across full images
    max_offset = int(0.25 * im_size)    
    x_scale_factor = x_size / max_offset
    y_size = 64.0   # display convenience only
    try:
        max_val = int(1.5 * np.amax(offsets))
        y_scale_factor = y_size / max_val
    except ZeroDivisionError:
        y_scale_factor = y_size
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

def _plot_local_keypoint_histogram(kp_histogram_forward,kp_histogram_backward,im_size):
    # Returns a histogram plot of number of local keypoints
    # Illustrates how many keypoints are local to each unmatched keypoint

    x_size = 1.0 * im_size    # stretch across one image
    max_kps = 128.0           # aribtrary
    x_scale_factor = x_size / max_kps
    y_size = 64.0               # for convenience of display
    max_val = int(1.5 * np.amax(np.concatenate((kp_histogram_forward,kp_histogram_backward),axis=0)))
    try: 
        y_scale_factor = y_size / max_val
    except ZeroDivisionError:
        y_scale_factor = y_size
    plot_box = 255 * np.ones((y_size,x_size,3),np.uint8)
    for n,val in enumerate(kp_histogram_forward):
        xpos = int(x_scale_factor * n)
        ypos = int(y_size-1-y_scale_factor*val)
        plot_box[ypos:y_size,xpos:xpos+1,:] = change_color_forward
    for n,val in enumerate(kp_histogram_backward):
        xpos = int(x_scale_factor * n)
        ypos = int(y_size-1-y_scale_factor*val)
        plot_box[ypos:y_size,xpos+2:xpos+3,:] = change_color_backward
    #x_cutoff = int(x_scale_factor*MIN_NEIGHBORHOOD_KEYPOINTS+4)
    #plot_box[:,x_cutoff,:] = (255,0,0) # mark MIN_NEIGHBORHOOD_KEYPOINTS
    mylabel = 'Local keypts.'#; MIN: {0}'.format(MIN_NEIGHBORHOOD_KEYPOINTS)
    cv2.putText(plot_box,mylabel,(20,16),1,1.0,black_color)
    plot_box[0,:,:] = (0,0,0)       # top divider
    plot_box[:,-1,:] = (0,0,0)      # edge divider
    return plot_box

def _poisson(x,a,y0,mu):
    # Defines a poisson distribution for curve fitting
    return y0 + a * poisson.pmf(x,mu)

def _calculate_proximity_threshold(offsets,MATCH_PROXIMITY_IN_PIXELS):
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


def _are_close(kpa,kpb,distance):
	# Returns true if keypoints are separated by less than distance

    a_coords = np.array([kpa.pt[0], kpa.pt[1]])
    b_coords = np.array([kpb.pt[0], kpb.pt[1]])
    dist = np.linalg.norm(a_coords - b_coords)
    if dist < distance:
        return True
    else:
        return False

def _calculate_homography(matches,kp1,kp2):
    """Determine the optimal homography transform between matches
    using a RANSAC-based estimator.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt
                              for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt
                              for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return H, mask

def _homography_distance(pt1,pt2,H):
    """After applying homography transform H to pt1, determine its
    distance from pt2.
    """ 
    # augmented coordinates
    aug1 = np.array((pt1[0],pt1[1],1))
    aug2 = np.array((pt2[0],pt2[1],1))
    dist = np.linalg.norm(np.dot(H,aug1) - aug2)
    return dist

def _compute_distance(pt1,pt2):
    """Compute the L2 distance between pt1 and pt2."""
    dist = np.linalg.norm(np.array(pt1)-np.array(pt2))
    return dist

def _check_homography(pt1,pt2,H,radius):
    """Determine whether the homography transform H on pt1 is within
    the given radius of pt2.
    """
    dist = _homography_distance(pt1,pt2,H)
    if dist < radius:
        return True
    else:
        return False
    
def detect_change(im1_file, im2_file, RelDir, output_dir,**kwparams):
    """Detect change between two input images."""

    # to comply with legacy variable naming
    FEATURES = kwparams['FEATURES']
    KAZE_PARAMETER = kwparams['KAZE_PARAMETER']
    KNEAREST = kwparams['KNEAREST']
    MATCH_PROXIMITY_IN_PIXELS = kwparams['MATCH_PROXIMITY_IN_PIXELS']
    CALCULATE_PROXIMITY_LIMIT = kwparams['CALCULATE_PROXIMITY_LIMIT']
    HOMOGRAPHY = kwparams['HOMOGRAPHY']
    MATCH_NEIGHBORHOOD_IN_PIXELS = kwparams['MATCH_NEIGHBORHOOD_IN_PIXELS']
    MATCH_PROBABILITY_THRESHOLD = kwparams['MATCH_PROBABILITY_THRESHOLD']
    STATS_TEST = kwparams['STATS_TEST']
    #CROSS_CHECK = kwparams['CROSS_CHECK']
    TRUE_CROSS_CHECK = kwparams['TRUE_CROSS_CHECK']
    #LONGER_MATCHLIST = kwparams['LONGER_MATCHLIST']

    # load image files [note grayscale: 0; color: 1]
    im1_file_relpath = os.path.join(RelDir, im1_file)
    im1 = cv2.imread(im1_file_relpath,0)
    im1_color = cv2.imread(im1_file_relpath,1)
    im2_file_relpath = os.path.join(RelDir, im2_file)
    im2 = cv2.imread(im2_file_relpath,0)
    im2_color = cv2.imread(im2_file_relpath,1)
    IMAGE_WIDTH = im1.shape[1]
    #file1 = im1_file.split(RelDir+'/')[1]
    #file2 = im2_file.split(RelDir+'/')[1]
    file_base = im2_file.split('-')[0]
    date1, date2 = bulk_wrapper.get_dates(im1_file,im2_file)
    save_image_filename = bulk_wrapper.generate_save_name(
        file_base+'-'+date1+date2,**kwparams)
    save_image_filename += '.jpg'

    # instantiate global OpenCV objects
    BFMATCH = cv2.BFMatcher()
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
#index_parameters = dict(algorithm = FLANN_KDTREE_INDEX, trees = FLANN_TREE_NUMBER)
#search_parameters = dict(checks=FLANN_SEARCH_DEPTH)
#FLANN = cv2.FlannBasedMatcher(index_parameters,search_parameters)

# find 2-way matches then do proximity test
#match_candidates = FLANN.match(desc1,desc2)
#match_candidates = BFMATCH.match(desc1,desc2)
# Use this instead for relaxed matching:
    match_candidates = BFMATCH.knnMatch(desc1,desc2,k=KNEAREST)
    print 'Found {0} match candidates...'.format(len(match_candidates))

    # calculate proximity limit
    top_match_candidates = [kNNlist[0] for kNNlist in match_candidates]
    offsets = _calculate_offsets_between_matches(kps1,kps2,
                                            top_match_candidates,IMAGE_WIDTH)
    if CALCULATE_PROXIMITY_LIMIT == True:
        proximity_limit = _calculate_proximity_threshold(offsets,
                                MATCH_PROXIMITY_IN_PIXELS)[0]
    else:
        proximity_limit = MATCH_PROXIMITY_IN_PIXELS
# 
    matches = []
    matches_forward = []
    matches_backward = []

    if HOMOGRAPHY:
        H, mask = _calculate_homography(top_match_candidates,kps1,kps2)
        print 'Estimated homography transformation:'
        print H
        for knnlist in match_candidates:
            for m in knnlist:
                if _check_homography(kps1[m.queryIdx].pt,kps2[m.trainIdx].pt,
                                     H,radius=proximity_limit):
                    matches_forward.append(m)
                    break
        if len(matches_forward) == 0:
            return [], (len(kps1)+len(kps2))/2.
    else:
        for knnlist in match_candidates:
            for m in knnlist:
                if _are_close(kps1[m.queryIdx],kps2[m.trainIdx],
                      proximity_limit):
                    matches_forward.append(m)
                    break

    # N.B. currently CROSS_CHECK is not set up to work with HOMOGRAPHY
    """
    if CROSS_CHECK:
        match_candidates_backwards = BFMATCH.knnMatch(desc2,desc1,k=KNEAREST)
        for knnlist in match_candidates_backwards:
            for m in knnlist:
                if _are_close(kps1[m.trainIdx],kps2[m.queryIdx],
                      proximity_limit):
                    matches_backward.append(m)
                    break
        if len(matches_forward) > len(matches_backward):
            for mf in matches_forward:
                for mb in matches_backward:
                    if mf.trainIdx == mb.queryIdx:
                        matches.append(mf)
                        #matches_backward.remove(mb)    # unique matching only
                        break
        else:
            for mb in matches_backward:
                for mf in matches_forward:
                    if mb.trainIdx == mf.queryIdx:
                        matches.append(mb)
                        break
    elif LONGER_MATCHLIST:
        match_candidates_backwards = BFMATCH.knnMatch(desc2,desc1,k=KNEAREST)
        for knnlist in match_candidates_backwards:
            for m in knnlist:
                if _are_close(kps1[m.trainIdx],kps2[m.queryIdx],
                      proximity_limit):
                    matches_backward.append(m)
                    break
        if len(matches_forward) > len(matches_backward):
            matches = list(matches_forward)
        else:
            matches = list(matches_backward)
    """        
    if TRUE_CROSS_CHECK:
        match_candidates_backwards = BFMATCH.knnMatch(desc2,desc1,k=KNEAREST)
        for knnlist in match_candidates_backwards:
            for m in knnlist:
                if _are_close(kps1[m.trainIdx],kps2[m.queryIdx],
                      proximity_limit):
                    matches_backward.append(m)
                    break
        for mf in matches_forward:
            for mb in matches_backward:
                if mf.trainIdx == mb.queryIdx and mf.queryIdx == mb.trainIdx:
                    matches.append(mf)
                    matches_backward.remove(mb)   
                    break
        """
        with open('kNNtruematch.log','a') as f:
            f.write(save_image_filename+':\n')
            f.write('lenkps1, lenkps2: {}, {}\n'.format(
                len(kps1),len(kps2)))
            f.write('lenmf, lenmb, lenmatches: {}, {}, {}\n'.format(
                len(matches_forward),len(matches_backward),len(matches)))
            try:
                rat = len(matches)/float(min(
                    len(matches_forward),len(matches_backward)))
            except ZeroDivisionError:
                rat = None
            f.write('matches/min(mf,mb): {}\n'.format(rat))
        return rat, None
        """
    else:
        matches = list(matches_forward)
                
    print '...of which {0} are within the proximity limit of {1} pixels.'.format(len(matches),proximity_limit)
    """
    if len(matches_forward) > len(matches_backward):
        kps1_matched = [kps1[m.queryIdx] for m in matches]
        kps2_matched = [kps2[m.trainIdx] for m in matches]
    else:
        kps1_matched = [kps1[m.trainIdx] for m in matches]
        kps2_matched = [kps2[m.queryIdx] for m in matches]
    """
    kps1_matched = [kps1[m.queryIdx] for m in matches]
    kps2_matched = [kps2[m.trainIdx] for m in matches]
    # calculate average match rate for each image
    N_kps1 = len(kps1)
    N_kps2 = len(kps2)
    N_matches = len(matches)
#match_rate_1 = N_matches/float(N_kps1)      #  average match rate for im1
#match_rate_2 = N_matches/float(N_kps2)       #  average match rate for im2

# examine neighborhood of each unmatched keypoint, count matched/un-matched keypoints
# first do forward direction
    kps2_not_matched = set(kps2) - set(kps2_matched)
    kps2_changed = []
    kps2_local_histogram = np.zeros((N_kps2,),dtype=np.int)

    for kp in kps2_not_matched:
        local_kps = filter(lambda x: _are_close(kp,x,MATCH_NEIGHBORHOOD_IN_PIXELS),kps2)
        local_matches = filter(lambda x: _are_close(kp,x,MATCH_NEIGHBORHOOD_IN_PIXELS),kps2_matched)
        kps2_local_histogram[len(local_kps)] += 1    

    # do statistical test for each un-matched keypoint
        if STATS_TEST == 'PBYKPRATE':
            local_kp_rate = len(local_kps)/float(N_kps2)
            if scipy.stats.binom.cdf(len(local_matches),N_matches,
                        local_kp_rate) < MATCH_PROBABILITY_THRESHOLD:
                kps2_changed.append(kp)
        elif STATS_TEST == 'MATCHKPRATIO':
            if scipy.stats.binom.cdf(len(local_matches),len(local_kps),
                                 match_rate_2) < MATCH_PROBABILITY_THRESHOLD:
                kps2_changed.append(kp)
        elif STATS_TEST == 'PBYMATCHRATE':
            local_match_rate = len(local_matches)/float(N_matches)
            if scipy.stats.binom.sf(len(local_kps),N_kps2,
                        local_match_rate) < MATCH_PROBABILITY_THRESHOLD:
                kps2_changed.append(kp)

    print 'Found {0} change keypoints (forward direction)'.format(len(kps2_changed))

# repeat above for backward direction
    kps1_not_matched = set(kps1) - set(kps1_matched)
    kps1_changed = []
    kps1_local_histogram = np.zeros((N_kps1,),dtype=np.int)

    for kp in kps1_not_matched:
        local_kps = filter(lambda x: _are_close(kp,x,MATCH_NEIGHBORHOOD_IN_PIXELS),kps1)
        local_matches = filter(lambda x: _are_close(kp,x,MATCH_NEIGHBORHOOD_IN_PIXELS),kps1_matched)
        kps1_local_histogram[len(local_kps)] += 1    

    # do statistical test for each un-matched keypoint
        if STATS_TEST == 'PBYKPRATE':
            local_kp_rate = len(local_kps)/float(N_kps1)
            if scipy.stats.binom.cdf(len(local_matches),N_matches,
                        local_kp_rate) < MATCH_PROBABILITY_THRESHOLD:
                kps1_changed.append(kp)
        elif STATS_TEST == 'MATCHKPRATIO':
            if scipy.stats.binom.cdf(len(local_matches),len(local_kps),
                                 match_rate_1) < MATCH_PROBABILITY_THRESHOLD:
                kps1_changed.append(kp)
        elif STATS_TEST == 'PBYMATCHRATE':
            local_match_rate = len(local_matches)/float(N_matches)
            if scipy.stats.binom.sf(len(local_kps),N_kps1,
                        local_match_rate) < MATCH_PROBABILITY_THRESHOLD:
                kps1_changed.append(kp)
                
    print 'Found {0} change keypoints (backward direction)'.format(len(kps1_changed))

    # prepare output image
    im1_out = im1_color.copy()
    im2_out = im2_color.copy()
    cv2.drawKeypoints(im2_color,kps2_changed,im2_out,
                      color=change_color_forward,flags=0)
    cv2.drawKeypoints(im2_out,kps1_changed,im2_out,
                      color=change_color_backward,flags=0)

# calculate histogram and plot
#match_offsets = _calculate_offsets_between_matches(kps1,kps2,match_candidates)
    offsets_histogram = _make_offsets_histogram(offsets,proximity_limit,
                                                IMAGE_WIDTH)
    local_keypoint_histogram = _plot_local_keypoint_histogram(kps2_local_histogram,kps1_local_histogram,IMAGE_WIDTH)
    hist_plots = np.concatenate((offsets_histogram,local_keypoint_histogram),axis=1)
    #hist_plots = np.concatenate((offsets_histogram,
     #                        255*np.ones(offsets_histogram.shape)),axis=1)

# make label box
    text_box_height = 10 + 20 * 7
    text_box = 255 * np.ones((text_box_height,2*IMAGE_WIDTH,3),np.uint8)

    label_string = date1
    label_origin = (20,20)
    cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
    label_string = date2
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
    cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
    label_string = "Changed keypoints:"
    label_origin = (20,80)
    cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
    label_string = "{0} forward".format(len(kps2_changed))
    label_origin = (200,80)
    cv2.putText(text_box,label_string,label_origin,1,1.0,change_color_forward)
    label_string = "{0} backward".format(len(kps1_changed))
    label_origin = (320,80)
    cv2.putText(text_box,label_string,label_origin,1,1.0,change_color_backward)
    label_string = "MATCH_NEIGHBORHOOD_PIXELS = {0}".format(MATCH_NEIGHBORHOOD_IN_PIXELS)
    label_origin = (20,100)
    cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
    label_string = "MATCH_PROBABILITY_THRESHOLD = {0}".format(MATCH_PROBABILITY_THRESHOLD)
    label_origin = (20,120)
    cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
    label_string = FEATURES + "; proximity test @ {0} pixels; BFMATCH".format(proximity_limit)
    label_origin = (20,140)
    cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)

    # join label to bottom of image pair and save
    im_A = np.concatenate((im1_color,im2_out),axis=1)
    im_B = np.concatenate((im_A,text_box),axis=0)
    im_C = np.concatenate((im_B,hist_plots),axis=0)
    print save_image_filename
    cv2.imwrite(os.path.join(output_dir,save_image_filename),im_C)

    return kps1_changed + kps2_changed, (len(kps1)+len(kps2))/2.

if __name__ == '__main__':
    """Needs updating: See syntax in bulk_wrapper.py"""
    """Ex: python bulk_detect.py 'dim1000test/CCclearcutsdim1000-2010.jpg'
       'dim1000test/CCclearcutsdim1000-2012.jpg'
    """
    im1_file = os.path.basename(sys.argv[1])
    im2_file = os.path.basename(sys.argv[2])
    dir_name = os.path.dirname(sys.argv[2])
    detect_change(im1_file,im2_file,dir_name,
                  **bulk_wrapper.default_change_params)
    

