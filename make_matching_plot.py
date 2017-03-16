""" make_matching_plot.py
Make plot of matching rate vs number of keypoints.
Arguments:
- log_file_1: Log file name 1
- log_file_2: Log file name 2
- log_file_3: Log file name 3
- save_image_filename: Save file name
"""

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

# main

# read args
parser = argparse.ArgumentParser(description='Make plot of matching rate.')
parser.add_argument('log_file_1', type=str, help='Log file 1.')
parser.add_argument('log_file_2', type=str, help='Log file 2.')
parser.add_argument('log_file_3', type=str, help='Log file 3.')
parser.add_argument('save_image_filename', type=str, help='Path to save output file.')
args = parser.parse_args()

kp_number_1 = np.genfromtxt(args.log_file_1, delimiter=',',usecols=3)
match_rate_1 = np.genfromtxt(args.log_file_1, delimiter=',',usecols=5)
kp_number_2 = np.genfromtxt(args.log_file_2, delimiter=',',usecols=3)
match_rate_2 = np.genfromtxt(args.log_file_2, delimiter=',',usecols=5)
kp_number_3 = np.genfromtxt(args.log_file_3, delimiter=',',usecols=3)
match_rate_3 = np.genfromtxt(args.log_file_3, delimiter=',',usecols=5)

plt.scatter(kp_number_1,match_rate_1,color='blue')
plt.scatter(kp_number_2,match_rate_2,color='green')
plt.scatter(kp_number_3,match_rate_3,color='red')
plt.xlim(0,10000)
plt.ylim(0,0.6)
plt.xlabel('Total keypoint number (both images)')
plt.ylabel('Match rate')

# save plot
plt.savefig(args.save_image_filename)

# show
plt.show()