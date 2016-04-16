"""Main function call for bulk runs of change detection.

Inputs:
Reads image files from specified directory (relative or absolute path).
Pairs are defined by having identical names up to the first '-'.

Currently only allows default parameters as specified in bulk_detect.py.

Output: A collection of image files, each of which includes an image pair,
change points, and parameters.

Example: $ python bulk_wrapper.py 'dim1000test/'
"""
import sys
import os
import itertools
import bulk_detect
import pdb

if __name__ == '__main__':
    CT_THRESHOLD_FRAC = .01
    CT_NBHD_SIZE = 120
    default_params = dict(FEATURES = 'KAZE',
                    KAZE_PARAMETER = 0.0003,
                    KNEAREST = 5,
                    MATCH_PROXIMITY_IN_PIXELS = 4,
                    CALCULATE_PROXIMITY_LIMIT = False,
                    MATCH_NEIGHBORHOOD_IN_PIXELS = 40,
                    MATCH_PROBABILITY_THRESHOLD = 1e-8)

    # WIP: forward default parameters
    params = default_params
    
    # read and pair the image files 
    try:
        image_dir = sys.argv[1]
    except IndexError:
        print 'Please specify a directory of image pairs.'
        sys.exit(0)
    fileSet = set()
    for dir_, _, files in os.walk(image_dir):
        relDir = os.path.relpath(dir_, os.getcwd())
        for fileName in files:
            #relDir = os.path.relpath(dir_, os.getcwd())
            #relFile = os.path.join(relDir, fileName)
            #fileSet.add(relFile)
            fileSet.add(fileName)
    image_files = sorted(fileSet)
    
    pairs = []
    for f1, f2 in itertools.izip(image_files[::2],image_files[1::2]):
        if f1.split('-')[0] == f2.split('-')[0]:
            pairs.append([f1,f2])
    for p in pairs:
        im2_color, changepoints, total_kps = bulk_detect.detect_change(*p,
                                                RelDir=relDir,**params)
        agg_image = bulk_detect.agglomerate(im2_color,changepoints,
                                CT_NBHD_SIZE,
                                total_kps*CT_THRESHOLD_FRAC)
        bulk_detect.write_agg_im(agg_image,CT_NBHD_SIZE,
                                total_kps*CT_THRESHOLD_FRAC,
                                *p,RelDir=relDir,**params)
    
