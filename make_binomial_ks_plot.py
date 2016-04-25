# make_binomial_ks_plot.py
"""
Make a plot of the Kolmogorov-Smirnov statistic for keypoint distribution vs binomial distribution.
Keypoints are KAZE, with statistic 0.0003. 
Arguments:
- log_file: Log file name with K-S scores
- save_file: Save file name
"""

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

# main

# read args
parser = argparse.ArgumentParser(description='Plot K-S statistic.')
parser.add_argument('log_file', type=str, help='Log file with K-S scores.')
parser.add_argument('save_file', type=str, help='Path to save output file.')
args = parser.parse_args()

kp_and_ks_numbers = np.genfromtxt(args.log_file, delimiter=',',usecols=(1,2))
kp_numbers = kp_and_ks_numbers[:,0]
ks_vals = kp_and_ks_numbers[:,1]
plt.scatter(kp_numbers,ks_vals)
plt.xlim(0,6000)
plt.ylim(0,0.6)
plt.xlabel('Number of KAZE keypoints detected')
plt.ylabel('K-S statistic vs. binomial distribution')

# save plot
plt.savefig(args.save_file)

# show
plt.show()