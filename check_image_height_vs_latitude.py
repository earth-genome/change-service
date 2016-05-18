# check_image_heigh_vs_latitude.py
"""
Check the height of NAIP imagery vs latitude.
Arguments:
- log_file: Log file of downloaded NAIP imagery
- save_file: File to save plot
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

# main

# read args
parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=str)
parser.add_argument('save_file', type=str)

args = parser.parse_args()

im_heights = []
im_lats = []

# check each entry in image log
with open(args.log_file, 'rU') as csvfile:
	datareader = csv.reader(csvfile)

	for row in datareader:
		# extract latitude from URL used to retrieve image
		im_url = row[1]
		m = re.search('lat=\d+\.\d+',im_url)
		if m:
			latstr = im_url[m.start()+4:m.end()]
			lat = float(latstr)
			im_lats.append(lat)
			# check image height in pixels
			im_file = row[0]
			im1 = cv2.imread(im_file,0)
			im_heights.append(im1.shape[0])
		else:
			print('No match on file {0}'.format(im_file))

# do linear fit
m,b = np.polyfit(im_lats,im_heights,1)
print("Linear fit: slope = {0}, offset = {1}".format(m,b))

xmin = 32
xmax = 40

fig = plt.figure(figsize=(9,5))
plt.scatter(im_lats,im_heights)	# plot data
plt.plot([xmin,xmax],[m*xmin+b,m*xmax+b],'r-')	# plot linear fit
plt.xlim(xmin,xmax)
plt.ylim(390,440)
plt.xlabel('Latitude')
plt.ylabel('Image height (pixels)')
plt.text(33,400,'Fit: {0} * Latitude + {1}'.format(m,b))

# save plot
plt.savefig(args.save_file)

# show
plt.show()