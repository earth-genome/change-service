"""Routines to check change detections against hand-labled imagery. 

Assumptions:

Points labeled as change have pixel values [255,0,255] (pink).

File names:
Image pair: 'UniqueName-Year1.jpg', 'UniqueName-Year2.jpg'
Labeled image: 'UniqueName-Year?labeled.png'
Binary change image: 'UniqueName-*.pgm'

Image directory:
For each UniqueName, there is exactly one pgm and one png file.

Usage from __main__: python validation.py 'change-mask-directory'
    'labeled_image_directory'
--> Outputs result to file 'change-labels-directory-valid.log'

External functions:

validate(change_mask,labeled_image)
tabulate(change_mask_directory,labeled_image_directory)


"""
import sys
import os
import itertools
import numpy as np
import cv2
import pdb

import bulk_wrapper

def validate(change_img,labeled_img):
    """Validate the change mask against a hand-labeled image.
    Assumes the labeled pixels take value [255,0,255]. Returns one of:
     'true positive', 'true negative', 'false positive','false negative',
     'missed detection' (for the case of positive detections that don't
     intersect any labeled regions).
    """

    PIXEL_MAX = 255
    PIXEL_MIN = 0
    PINK = [PIXEL_MAX,PIXEL_MIN,PIXEL_MAX]
    
    # convert images to binary
    change_bin = change_img/PIXEL_MAX
    label_bin = np.zeros(labeled_img.shape[:2])

    label_points = np.where(np.all(labeled_img==PINK,axis=2))
    label_bin[label_points] = 1
    
    label_max = np.max(label_bin)
    change_max = np.max(change_bin)
    if label_max == 0:
        if change_max == 0:
            result = 'true negative'
        else:
            result = 'false positive'
    else:
        if change_max == 0:
            result = 'false negative'
        elif np.max(change_bin*label_bin) == 1:
            result = 'true positive'
        else:
            result = 'erroneous detection'
    label = 'change' if label_max else 'no change'
    return result, label

def tabulate(change_dir,labeled_dir):
    """Run through directories of change detections and labeled images
    and count the numbers of true/false detections.
    """

    # read and pair the detection binaries with the labeled images
    change_files = bulk_wrapper.sort_file_names(change_dir, '.pgm')
    label_files = bulk_wrapper.sort_file_names(labeled_dir, 'labeled.png')
    if not change_files:
        sys.exit('No change files found.')
    if not label_files:
        sys.exit('No labeled image files found.')
        
    pairs = []
    for f1, f2 in itertools.izip(change_files,label_files):
        if (f1.split(bulk_wrapper.SPLIT_STRING)[0] ==
            f2.split(bulk_wrapper.SPLIT_STRING)[0]):
            pairs.append([f1,f2])
        else:
            sys.exit('Warning: {} or {} is missing a pair.'.format(f1,f2))

    # count and log the validation results
    logfile = open(change_dir+'/'+change_dir+'-valid.log','a')
    totals = {result:0 for result in ['true positive','true negative',
        'false positive','false negative', 'erroneous detection']}
    label_totals = {label:0 for label in ['no change','change']}
    for p in pairs:
        cf = cv2.imread(change_dir+'/'+p[0],0)
        lf = cv2.imread(labeled_dir+'/'+p[1],1)
        result, label = validate(cf,lf)
        totals[result] += 1
        label_totals[label] += 1
        logfile.write('Image {}: {}\n'.format(p[0],result))
    logfile.write('\nFrom {} images, we have: {}\n'.format(len(pairs),
                                                             totals))
    print 'From {} images, we have: {}\n'.format(len(pairs),totals)
    totalaccuracy = (totals['true positive']
                + totals['true negative'])/float(len(pairs))
    try: 
        posrate = totals['true positive']/float(label_totals['change'])
    except ZeroDivisionError:
        posrate = None
    try:
        negrate = totals['true negative']/float(label_totals['no change'])
    except ZeroDivisionError:
        negrate = None
    logfile.write('Total accuracy: {:.4f}, Correct positive rate: {}, Correct negative rate: {}\n'.format(totalaccuracy,posrate,negrate))
    print('Total accuracy: {:.4f}, Correct positive rate: {}, Correct negative rate: {}\n'.format(totalaccuracy,posrate,negrate))

    try:
        tpfpratio = totals['true positive']/float(
            totals['false positive']+totals['erroneous detection'])
    except ZeroDivisionError:
        tpfpratio = None
    logfile.write('Ratio of true to false positives: {}\n--- ---\n\n'.format(tpfpratio))
    print('Ratio of true to false positives: {}\n--- ---\n\n'.format(tpfpratio))
        
    logfile.close()
    return

def compute_stats(labeled_dir):
    """Compute and print average area and std dev of areas of
    labeled regions.
    """
    
    # Use the pink for labeled images, the white for change images
    PIXEL_MAX = 255
    PIXEL_MIN = 0
    PINK = [PIXEL_MAX,PIXEL_MIN,PIXEL_MAX]
    WHITE = [PIXEL_MAX,PIXEL_MAX,PIXEL_MAX]

    #COLOR = PINK
    #FILE_EX = 'labeled.png'

    COLOR = WHITE
    FILE_EX = '.pgm'

    imagenames = []
    for dir_, _, files in os.walk(labeled_dir):
        for filename in files:
            if FILE_EX in filename:
                imagenames.append(filename)
                
    areas = []
    for i in imagenames:
        labeled_img = cv2.imread(labeled_dir+'/'+i,1)
        # convert img to binary
        label_bin = np.zeros(labeled_img.shape[:2])
        label_points = np.where(np.all(labeled_img==COLOR,axis=2))
        label_bin[label_points] = 1

        # only images with change
        change_area = np.sum(label_bin)
        tot_area = reduce(lambda x,y: x*y, label_bin.shape)
        if change_area > 0:
            frac_area = change_area/float(tot_area)
            areas.append(frac_area)
            print i
            print 'Frac area: {:4f}'.format(frac_area)

    if len(areas) == 0:
        print 'nothing to see here'
        return
    print '\nFor {} images with change,\n' \
        'the average fractional area with change is {:.4f},\n' \
        'the std dev is {:4f},\n' \
        'the min is {:4f}, and the max is {:4f}.'.format(len(areas),
        np.mean(areas),np.std(areas),np.min(areas),np.max(areas))
    return
        
        

if __name__ == '__main__':
    # last-minute hack for statistical analysis: Usage:
    # python validation.py 'construction-labeled' 'stats'
    if sys.argv[2] == 'stats':
        compute_stats(sys.argv[1])
        sys.exit(1)

    # As was:
    try:
        change_dir, labeled_dir = sys.argv[1:3]
    except ValueError:
        sys.exit('Pls. specify directories of change masks ' +
                 'and labeled images.')
    
    tabulate(change_dir,labeled_dir)
    
    
