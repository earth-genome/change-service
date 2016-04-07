"""Main function call for bulk runs of change detection.

Inputs:
Reads image files from specified directory (relative or absolute path).
Pairs are defined by having identical names up to the first '-'.

Currently only allows default parameters as specified in bulk_detect.py.

Output: A collection of image files, each of which includes an image pair,
change points, and parameters.
"""
import sys
import os
import itertools
import bulk_detect
import pdb

if __name__ == '__main__':

    

    # read and pair the image files 
    try:
        image_dir = sys.argv[1]
    except IndexError:
        print 'Please specify a directory of image pairs.'
        sys.exit(0)
    fileSet = set()
    for dir_, _, files in os.walk(image_dir):
        for fileName in files:
            relDir = os.path.relpath(dir_, os.getcwd())
            relFile = os.path.join(relDir, fileName)
            fileSet.add(relFile)
    image_files = sorted(fileSet)
    pairs = []
    for f1, f2 in itertools.izip(image_files[::2],image_files[1::2]):
        if f1.split('-')[0] == f2.split('-')[0]:
            pairs.append([f1,f2])
    for p in pairs:
        bulk_detect.detect_change(*p,relDir)
    
