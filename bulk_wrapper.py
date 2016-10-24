"""Main function call for bulk runs of change detection.

Inputs:
1) Paired image files from specified directory (relative or absolute path).
Pairs are defined by having identical names up to the SPLIT_STRING (defined
below).
2) Optionally, change-labeled images (file format *labeled.png) from another directory.
** Parameters are currently hard set in PARAMETERS, top of this file

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
usage: bulk_wrapper.py [-h] [-l LABELED_DIR] [-t Prob. Threshold] pairs_dir

Detect change between image pairs.

positional arguments:
  pairs_dir             directory containing image pairs - no trailing /

optional arguments:
  -h, --help            show this help message and exit
  -l LABELED_DIR, --labeled_dir LABELED_DIR
                        directory containing labeled images
  -t 1e-xx, match probability threshold

Set parameters by editing defaults, top of bulk_wrapper.py.

Examples:
1) $ python bulk_wrapper.py 'dim1000test/'
2) $ python bulk_wrapper.py 'dim1000practice'
        -l 'dim1000practice-labeled'
        -t 1e-08
"""

import sys
import os
import itertools
import argparse
import numpy as np
import scipy.signal
import cv2
import time

import bulk_detect
import validation

# defaults as of 5/1/16
#default_params = dict(FEATURES = 'KAZE',
#                    KAZE_PARAMETER = 0.0003,
#                    KNEAREST = 5,
#                    MATCH_PROXIMITY_IN_PIXELS = 4,
#                    CALCULATE_PROXIMITY_LIMIT = False,
#                    TRUE_CROSS_CHECK = True,
#                    HOMOGRAPHY = False,
#                    MATCH_NEIGHBORHOOD_IN_PIXELS = 40,
#                    STATS_TEST = 'PBYKPRATE',
#                    MATCH_PROBABILITY_THRESHOLD = 1e-8,
#                    CT_THRESHOLD_FRAC = .01,
#                    CT_NBHD_SIZE = 120)
PARAMETERS = dict(FEATURES = 'KAZE',
                    KAZE_PARAMETER = 0.0003,
                    KNEAREST = 5,
                    MATCH_PROXIMITY_IN_PIXELS = 4,
                    CALCULATE_PROXIMITY_LIMIT = False,
                    TRUE_CROSS_CHECK = True,
                    HOMOGRAPHY = False,
                    MATCH_NEIGHBORHOOD_IN_PIXELS = 30,
                    STATS_TEST = 'PBYKPRATE',
                    MATCH_PROBABILITY_THRESHOLD = 1e-8,
                    CT_THRESHOLD_FRAC = .01,
                    CT_NBHD_SIZE = 120)
#SPLIT_STRING = 'dim1000-'
SPLIT_STRING = '-'
INPUT_FILEEXT = '.png'
#INPUT_FILEEXT = '.jpg'

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
    return [f.split(SPLIT_STRING)[1].split('.')[0] for f in filenames]
               
def generate_save_name(base_name,**params):
    """Generate a string recording base_name and key parameters."""
    save_name = '-'.join([base_name,
                params['FEATURES'],
                #'kNN'+str(params['KNEAREST']),
                #'prox'+str(params['MATCH_PROXIMITY_IN_PIXELS']),
        'thr{:.0e}'.format(params['MATCH_PROBABILITY_THRESHOLD'])])
    if params['KAZE_PARAMETER'] != 0.0003:
        save_name += 'KAZEpar'+str(params['KAZE_PARAMETER'])
    if params['MATCH_NEIGHBORHOOD_IN_PIXELS'] != 40:
        save_name += ('nbhd'+
                      str(params['MATCH_NEIGHBORHOOD_IN_PIXELS']))
    if params['CALCULATE_PROXIMITY_LIMIT'] == True:
        save_name += 'calcproxlim'
    if params['HOMOGRAPHY'] == True:
        save_name += 'homog'
    if params['STATS_TEST'] != 'PBYKPRATE':
        save_name += params['STATS_TEST']
    #if params['CROSS_CHECK']:
    #    save_name += 'XCheck'
    #if params['TRUE_CROSS_CHECK']:
    #    save_name += 'TrueXCheck'
    #if changeparams['LONGER_MATCHLIST']:
    #    save_name += 'LongerMList'
    if params['CT_THRESHOLD_FRAC'] != .01:
        save_name += '-ctfrac{}'.format(params['CT_THRESHOLD_FRAC'])
    if params['CT_NBHD_SIZE'] != 120:
        save_name += 'ctnbhd{:d}'.format(params['CT_NBHD_SIZE'])
    
    return save_name
    

def agglomerate(changepoints,total_kps,*file_data,**kwparams):
    """Count the number of changepoints on each nbhd_size square, and
    threshold the count by ct_threshold.  Writes to file a mask defined
    by whether the neighborhood centered on the point does or does not
    meet the ct_threshold, along with a triptych of the original images
    plus the second image alpha-masked.
    """

    OPACITY = 63 # alpha channel, on a scale 0-255
    FULL_OPACITY = 255
    WHITE = 255
    ct_nbhd_size = kwparams['CT_NBHD_SIZE']
    ct_threshold = total_kps*kwparams['CT_THRESHOLD_FRAC']
    im2_file = file_data[1]
    im2_color = cv2.imread(os.path.join(file_data[2], im2_file),1)
    try:
        imbb_color = cv2.imread(os.path.join(file_data[-1],
                    im2_file.split(INPUT_FILEEXT)[0]+'labeledbb.png'))
    except IOError:
        imbb_color = None
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
    if imbb_color is not None:
        aggbb_im = np.dstack((imbb_color,alpha_mask))
    else:
        aggbb_im = None
    
    agg_binary = np.zeros(image_shape,dtype=int)
    agg_binary[ct_image > ct_threshold] = WHITE
    write_agg_images(agg_im,agg_binary,imbb_color,aggbb_im,
                    *file_data,**kwparams)
    return 

def write_agg_images(agg_image,agg_bin,bb_image,aggbb_image,
                 im1_file,im2_file,RelDir,agg_dir,bb_dir,**params):
    """Write the agg image to file."""
    FULL_OPACITY = 255
    file_base = im2_file.split(SPLIT_STRING)[0]
    date1, date2 = get_dates(im1_file,im2_file)
    save_name = (generate_save_name(file_base+SPLIT_STRING
                                   +date1+date2,**params) + '-agged')
    im1_color = cv2.imread(os.path.join(RelDir, im1_file),1)
    im2_color = cv2.imread(os.path.join(RelDir, im2_file),1)
    blank_mask = np.ones(im2_color.shape[:2],dtype=int)*FULL_OPACITY
    im1_masked = np.dstack((im1_color,blank_mask))
    im2_masked = np.dstack((im2_color,blank_mask))
    triptych = np.hstack((im1_masked,im2_masked,agg_image))
    cv2.imwrite(os.path.join(agg_dir,save_name+'.png'),triptych)
    cv2.imwrite(os.path.join(agg_dir,save_name+'.pgm'),agg_bin)
    if bb_image is not None:
        bb_masked = np.dstack((bb_image,blank_mask))
        triptychbb = np.hstack((im1_masked,bb_masked,agg_image))
        triptychaggbb = np.hstack((im1_masked,im2_masked,aggbb_image))
        cv2.imwrite(os.path.join(agg_dir,save_name+'bb.png'),triptychbb)
        cv2.imwrite(os.path.join(agg_dir,save_name+'bbagg.png'),
                     triptychaggbb)
    return

if __name__ == '__main__':
    params = PARAMETERS
    parser = argparse.ArgumentParser(
        description='Detect change between image pairs.',
        epilog='Set parameters by editing defaults, top of bulk_wrapper.py.')
    parser.add_argument('pairs_dir', type=str,
                    help='directory containing image pairs - no trailing /')
    parser.add_argument('-l', '--labeled_dir', type=str,
                        help='directory containing labeled images')
    parser.add_argument('-t','--thresh', type=float,
                        help='Match probability threshold.')
    # WIP: For now adjust PARAMETERS above
    """
    parser.add_argument('--FEATURES', action='store_const',
                        const='SIFT', default='KAZE',
                        help='SIFT or KAZE')
    parser.add_argument('--KAZE_PARAMETER', default=.0003)
    parser.add_argument('--KNEAREST', default=5)
    parser.add_argument('--MATCH_PROXIMITY_IN_PIXELS', default=4)
    parser.add_argument('--CALCULATE_PROXIMITY_LIMIT', action='store_true')
    parser.add_argument('--MATCH_NEIGHBORHOOD_IN_PIXELS', default=40)
    parser.add_argument('--CT_THRESHOLD_FRAC', default=.01)
    parser.add_argument('--CT_NBHD_SIZE', default=120)
    """
    args = parser.parse_args()
    if args.thresh is not None:
        params['MATCH_PROBABILITY_THRESHOLD'] = args.thresh

    image_files = sort_file_names(args.pairs_dir)
    pairs_basename = os.path.basename(args.pairs_dir)
    output_dir = (generate_save_name(pairs_basename,**params) +'-changepts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    agg_dir = (generate_save_name(pairs_basename,**params) + '-agged')
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)
    with open(agg_dir+'/'+agg_dir+'-valid.log','a') as logfile:
        logfile.write('bulk_wrapper.py {}.\n'.format(time.strftime('%c')))
        logfile.write('Change parameters:\n')
        for k,v in params.iteritems():
            logfile.write('{}: {}\n'.format(k,v))
        logfile.write('\n')

    pairs = []
    for f1, f2 in itertools.izip(image_files[::2],image_files[1::2]):
        if f1.split(SPLIT_STRING)[0] == f2.split(SPLIT_STRING)[0]:
            pairs.append([f1,f2])
    for p in pairs:
        file_data = [f for f in p] + [args.pairs_dir,output_dir] 
        changepoints, total_kps = bulk_detect.detect_change(*file_data,
                                                **params)
        file_data.pop()
        file_data += [agg_dir,args.labeled_dir+'bb']
        agglomerate(changepoints,total_kps,*file_data,**params)
    if args.labeled_dir is not None:
        validation.tabulate(agg_dir,args.labeled_dir)

        
    
