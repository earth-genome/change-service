""" compare_keypoints.py
Compare SIFT and KAZE keypoints between two images.
Arguments:
- filename1: Image file
- filename2: Filename to save results
- filename3: Log file name
"""

import argparse
import numpy as np
import cv2

# global parameters
KAZE_PARAMETER = 0.001                      # default for KAZE
#KAZE_PARAMETER = 0.0003                    # empirical
IMAGE_WIDTH = 512                           # expected image width

# main

# set up color choices
black_color = (0,0,0)
sift_color = (0,0,255)
kaze_color = (0,255,0)

# read args
parser = argparse.ArgumentParser(description='Compare SIFT and KAZE keypoints.')
parser.add_argument('image_filename', type=str, help='Image file name.')
parser.add_argument('save_image_filename', type=str, help='Path to save output file.')
parser.add_argument('log_file', type=str, help='Log file.')
args = parser.parse_args()

# load image files [note grayscale: 0; color: 1]
im = cv2.imread(args.image_filename,0)
im_color = cv2.imread(args.image_filename,1)

# instantiate global OpenCV objects
KAZE = cv2.KAZE_create(threshold = KAZE_PARAMETER)
#sift_features = {"nFeatures":0,"nOctaveLayers":3,"contrastThreshold":2,"edgeThreshold":2,"sigma":2}
SIFT = cv2.xfeatures2d.SIFT_create()        # use default features

# find keypints and descriptors
kps_kaze = KAZE.detect(im,None)
kps_sift = SIFT.detect(im,None)
print 'Found {0} KAZE kps'.format(len(kps_kaze))
print 'Found {0} SIFT kps'.format(len(kps_sift))

# OUTPUT
# log keypoint values
f_log = open(args.log_file,'a')
log_string = "{0}, {1}, {2}, {3}\n".format(args.image_filename,KAZE_PARAMETER,len(kps_kaze),len(kps_sift))
f_log.write(log_string)
f_log.close()
		
# build output image
im_out = im_color.copy()
cv2.drawKeypoints(im_out,kps_kaze,im_out,color=kaze_color,flags=0)
cv2.drawKeypoints(im_out,kps_sift,im_out,color=sift_color,flags=0)

# make label box
text_box_height = 10 + 20 * 3
text_box = 255 * np.ones((text_box_height,IMAGE_WIDTH,3),np.uint8)
label_string = args.image_filename
label_origin = (20,20)
cv2.putText(text_box,label_string,label_origin,1,1.0,black_color)
label_string = "SIFT keypoints: {0}".format(len(kps_sift))
label_origin = (20,40)
cv2.putText(text_box,label_string,label_origin,1,1.0,sift_color)
label_string = "KAZE keypoints: {0}".format(len(kps_kaze))
label_origin = (20,60)
cv2.putText(text_box,label_string,label_origin,1,1.0,kaze_color)
label_string = "KAZE threshold: {:1.4f}".format(KAZE_PARAMETER)
label_origin = (20+256,60)
cv2.putText(text_box,label_string,label_origin,1,1.0,kaze_color)

# join label to bottom of image pair and save
im_out = np.concatenate((im_out,text_box),axis=0)
cv2.imwrite(args.save_image_filename,im_out)

