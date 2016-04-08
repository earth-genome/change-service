#get_naip_imagery.py
"""
Get NAIP imagery.
:params:
: prefix_name: string prefix for imagery save filenames
: save_location: directory to save imagery
: log_file: name of file to log downloaded images
"""

import argparse
import urllib
import urllib2
import json
import os

# params
BASE_LAT, BASE_LONG = 37.314,-121.948				# san jose
#BASE_LONG = -121.948			# san jose
DELTA_LAT, DELTA_LONG = 0.02, 0.03
NUMBER_OF_STEPS = 4   	# will grab this many images squared times 2 (for 2010, 2012)

# api info
API_BASE_IMAGERY = 'http://waterapp.enviro-service.appspot.com/imagery/naip'

# read args
parser = argparse.ArgumentParser(description='Download NAIP imagery.')
parser.add_argument('prefix_name', type=str, help='Prefix for save image filename.')
parser.add_argument('save_location', type=str, help='Directory to save imagery.')
parser.add_argument('log_file', type=str, help='Filename to log images downloaded.')
args = parser.parse_args()

# build payload and make call for imagery

# make save directory if necessary
if not os.path.exists(args.save_location):
	os.makedirs(args.save_location)

# open log file
f_log = open(args.log_file,'a')

for i in range(NUMBER_OF_STEPS):
	mylat = BASE_LAT + i * DELTA_LAT
	for j in range(NUMBER_OF_STEPS):
		num = i * NUMBER_OF_STEPS + j
		mylong = BASE_LONG + j * DELTA_LONG
		payload = {}
		payload['lon'] = mylong
		payload['lat'] = mylat
		payload['dimension'] = 200
		payload['year'] = 2010			# do 2010 first
		url_payload = urllib.urlencode(payload)
		full_url = API_BASE_IMAGERY + '?' + url_payload
		response = urllib2.urlopen(full_url)
		image = response.read()
		filename = "{}{:0>3d}-2010.jpg".format(args.prefix_name,num)	# pad num with zeros to 3 digits
		save_path = os.path.join(args.save_location,filename)
		with open(save_path,'wb') as f:
			f.write(image)
			log_string = "{0}, {1}\n".format(save_path,full_url)
			f_log.write(log_string)
			print 'Saved image to file: {0} from URL: {1}'.format(save_path,full_url)
			f.close()
		payload['year'] = 2012		# now do 2012
		url_payload = urllib.urlencode(payload)
		full_url = API_BASE_IMAGERY + '?' + url_payload
		response = urllib2.urlopen(full_url)
		image = response.read()
		filename = "{}{:0>3d}-2012.jpg".format(args.prefix_name,num)	# pad num with zeros to 3 digits
		save_path = os.path.join(args.save_location,filename)
		with open(save_path,'wb') as f:
			f.write(image)
			log_string = "{0}, {1}\n".format(save_path,full_url)
			f_log.write(log_string)
			print 'Saved image to file: {0} from URL: {1}'.format(save_path,full_url)
			f.close()

f_log.close()
print 'Finished'

