"""Routines to check change detections against hand-labled imagery. 

Assumptions:

Points labeled as change have pixel values [255,255,255] (white).

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
    Assumes the labeled pixels take value [255,255,255]. Returns one of:
     'true positive', 'true negative', 'false positive','false negative',
     'missed detection' (for the case of positive detections that don't
     intersect any labeled regions).
    """

    # WIP: Currently a true positive is given if any labeled change region
    # intersects with any detection region in the change mask.  This could
    # be improved to check multiple regions independently.

    WHITE = 255
    
    # convert images to binary
    change_bin = change_img/WHITE
    label_bin = np.zeros(labeled_img.shape[:2])
    label_bin[np.where(np.all(labeled_img==[255,255,255],axis=2))] = 1

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
        if f1.split('-')[0] == f2.split('-')[0]:
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
        tpfpratio = totals['true positive']/float(totals['false positive'])
    except ZeroDivisionError:
        tpfpratio = None
    logfile.write('Ratio of true to false positives: {}\n--- ---\n\n'.format(tpfpratio))
    print('Ratio of true to false positives: {}\n--- ---\n\n'.format(tpfpratio))
        
    logfile.close()
    return

if __name__ == '__main__':
    try:
        change_dir, labeled_dir = sys.argv[1:3]
    except ValueError:
        sys.exit('Pls. specify directories of change masks ' +
                 'and labeled images.')
    
    tabulate(change_dir,labeled_dir)
