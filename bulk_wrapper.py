"""Main function call for bulk runs of change detection.

Inputs:
1) Paired image files from specified directory (relative or absolute path).
Pairs are defined by having identical names up to the first '-'.
2) Optionally, change-labeled images (file format *labeled.png) from another directory.
** Parameters are currently hard set in default_change_params and
default_agg_params, top of this file. 

Outputs:
1) A collection of image files, each of which includes an image pair,
change points, and parameters.
2) A collection of change masks, derived by aggregating change points,
and a collection of image triptyches with the original images beside
the second image with an alpha change mask.
3) If a directory with labeled images is specified, a log file with results
of validation of the change masks against image labels.

Help:
$python bulk_wrapper.py -h
usage: bulk_wrapper.py [-h] [--labeled_dir LABELED_DIR] pairs_dir

Detect change between image pairs.

positional arguments:
  pairs_dir             Directory containing image pairs.

optional arguments:
  -h, --help            show this help message and exit
  -l, --labeled_dir LABELED_DIR
                        Directory containing labeled images.

Set parameters by editing defaults, top of bulk_wrapper.py

Examples:
1) $ python bulk_wrapper.py 'dim1000test/'
2) $ python bulk_wrapper.py 'dim1000practice' 'dim1000practice-labeled'
"""

import sys
import os
import itertools
import argparse
import numpy as np
import scipy.signal
import cv2

import bulk_detect
import validation

# defaults as of 5/1/16
default_change_params = dict(FEATURES = 'KAZE',
                    KAZE_PARAMETER = 0.0003,
                    KNEAREST = 5,
                    MATCH_PROXIMITY_IN_PIXELS = 4,
                    CALCULATE_PROXIMITY_LIMIT = False,
                    MATCH_NEIGHBORHOOD_IN_PIXELS = 40,
                    MATCH_PROBABILITY_THRESHOLD = 1e-8)
default_agg_params = dict(CT_THRESHOLD_FRAC = .01,
                      CT_NBHD_SIZE = 120)

def sort_file_names(directory, textstring=None):
    """Parse and alphabetically sort file names from directory that contain
    the text string.  Returns a sorted list of file names.
    """
    fileSet = set()
    for dir_, _, files in os.walk(directory):
        if textstring is not None:
            for fileName in files:
                if textstring in fileName:
                    fileSet.add(fileName)
        else:
            for fileName in files:
                fileSet.add(fileName)
    file_names = sorted(fileSet)
    return file_names

def get_dates(*filenames):
    """Extract dates from filenames of standard format."""
    return [f.split('-')[1].split('.jpg')[0] for f in filenames]
               
def generate_save_name(base_name,**params):
    """Generate a string recording base_name and key parameters."""
    save_name = '-'.join([base_name,
                        params['FEATURES'],
                        'KAZEpar'+str(params['KAZE_PARAMETER']),
                        'kNN'+str(params['KNEAREST']),
                        'prox'+str(params['MATCH_PROXIMITY_IN_PIXELS']),
                        'nbhd'+str(params['MATCH_NEIGHBORHOOD_IN_PIXELS']),
                        'thresh'+str(params['MATCH_PROBABILITY_THRESHOLD'])])
    return save_name
    
def agglomerate(changepoints,ct_nbhd_size,ct_threshold,
                *file_data,**kwparams):
    """Count the number of changepoints on each nbhd_size square, and
    threshold the count by ct_threshold.  Writes to file a mask defined
    by whether the neighborhood centered on the point does or does not
    meet the ct_threshold, along with a triptych of the original images
    plus the second image alpha-masked.
    """

    OPACITY = 63 # alpha channel, on a scale 0-255
    FULL_OPACITY = 255
    WHITE = 255
    im2_color = cv2.imread(os.path.join(file_data[2], file_data[1]),1)
    image_shape = im2_color.shape[:-1]
    changepoints_image = np.zeros(image_shape,dtype=int)
    ct_kernel = np.ones((ct_nbhd_size,ct_nbhd_size))    
    for kp in changepoints:
        changepoints_image[kp.pt[1],kp.pt[0]] = 1
    ct_image = scipy.signal.fftconvolve(changepoints_image,ct_kernel,
                                        mode='same')
    alpha_mask = np.ones(image_shape,dtype=int)*OPACITY
    alpha_mask[ct_image > ct_threshold] = FULL_OPACITY
    agg_im = np.dstack((im2_color,alpha_mask))
    agg_binary = np.zeros(image_shape,dtype=int)
    agg_binary[ct_image > ct_threshold] = WHITE
    write_agg_images(agg_im,agg_binary,ct_nbhd_size,ct_threshold,
                     *file_data,**kwparams)
    return 

def write_agg_images(agg_image,agg_bin,ct_nbhd_size,ct_threshold,
                 im1_file,im2_file,RelDir,agg_dir,
                 **kwparams):
    """Write the agg image to file."""
    FULL_OPACITY = 255
    file_base = im2_file.split('-')[0]
    date1, date2 = get_dates(im1_file,im2_file)
    save_name = generate_save_name(file_base+'-'+date1+date2,**kwparams)
    save_name = '-'.join([save_name,
                        'ct'+str(ct_threshold),
                        'ct_nbhd'+str(ct_nbhd_size),
                        'agged'])
    im1_color = cv2.imread(os.path.join(RelDir, im1_file),1)
    im2_color = cv2.imread(os.path.join(RelDir, im2_file),1)
    blank_mask = np.ones(im2_color.shape[:2],dtype=int)*FULL_OPACITY
    im1_masked = np.dstack((im1_color,blank_mask))
    im2_masked = np.dstack((im2_color,blank_mask))
    triptych = np.hstack((im1_masked,im2_masked,agg_image))
    cv2.imwrite(os.path.join(agg_dir,save_name+'.png'),triptych)
    cv2.imwrite(os.path.join(agg_dir,save_name+'.pgm'),agg_bin)
    return

if __name__ == '__main__':
    change_params = default_change_params
    agg_params = default_agg_params

    parser = argparse.ArgumentParser(
        description='Detect change between image pairs.',
        epilog='Set parameters by editing defaults, top of bulk_wrapper.py')
    parser.add_argument('pairs_dir', type=str,
                        help='Directory containing image pairs.')
    parser.add_argument('-l', '--labeled_dir', type=str,
                        help='Directory containing labeled images.')

    # WIP: For now adjust default_change_params, default_agg_params above
    """
    parser.add_argument('--FEATURES', action='store_const',
                        const='SIFT', default='KAZE',
                        help='SIFT or KAZE')
    parser.add_argument('--KAZE_PARAMETER', default=.0003)
    parser.add_argument('--KNEAREST', default=5)
    parser.add_argument('--MATCH_PROXIMITY_IN_PIXELS', default=4)
    parser.add_argument('--CALCULATE_PROXIMITY_LIMIT', action='store_true')
    parser.add_argument('--MATCH_NEIGHBORHOOD_IN_PIXELS', default=40)
    parser.add_argument('--MATCH_PROBABILITY_THRESHOLD', default=1e-8)
    parser.add_argument('--CT_THRESHOLD_FRAC', default=.01)
    parser.add_argument('--CT_NBHD_SIZE', default=120)
    """
    args = parser.parse_args()
    image_files = sort_file_names(args.pairs_dir)
    output_dir = args.pairs_dir + '-changepts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    agg_dir = args.pairs_dir + '-agged'
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)

    pairs = []
    for f1, f2 in itertools.izip(image_files[::2],image_files[1::2]):
        if f1.split('-')[0] == f2.split('-')[0]:
            pairs.append([f1,f2])
    for p in pairs:
        file_data = [f for f in p] + [args.pairs_dir,output_dir] 
        changepoints, total_kps = bulk_detect.detect_change(*file_data,
                                                **change_params)
        file_data.pop()
        file_data += [agg_dir]
        agglomerate(changepoints,agg_params['CT_NBHD_SIZE'],
                                total_kps*agg_params['CT_THRESHOLD_FRAC'],
                                *file_data,**change_params)
    if args.labeled_dir is not None:
        validation.tabulate(agg_dir,args.labeled_dir)
        
        
        
    
