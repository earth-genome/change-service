""" make_keypoint_plot.py
Make plot of number of SIFT and KAZE keypoints detected in each image.
Arguments:
- filename1: Log file name
- filename2: Save file name
"""

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

# global parameters

# main

# set up color choices
black_color = (0,0,0)
sift_color = (0,0,255)
kaze_color = (0,255,0)

# read args
parser = argparse.ArgumentParser(description='Compare SIFT and KAZE keypoints.')
parser.add_argument('log_file', type=str, help='Log file.')
parser.add_argument('save_image_filename', type=str, help='Path to save output file.')
args = parser.parse_args()

detected_keypoint_numbers = np.genfromtxt(args.log_file, delimiter=',',usecols=(2,3))
kaze_vals = detected_keypoint_numbers[:,0]
sift_vals = detected_keypoint_numbers[:,1]
plt.scatter(kaze_vals,sift_vals)
plt.xlim(0,6000)
plt.ylim(0,6000)
plt.xlabel('KAZE keypoints')
plt.ylabel('SIFT keypoints')

# save plot
plt.savefig(args.save_image_filename)

# show
plt.show()