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
#BASE_LAT, BASE_LONG = 37.314,-121.948				# san jose
#BASE_LAT, BASE_LONG = 37.307, -120.477				# merced
#BASE_LAT, BASE_LONG = 33.802, -118.340				# los angeles
#BASE_LAT, BASE_LONG = 38.444, -123.125				# jenner
#BASE_LAT, BASE_LONG = 39.103, -121.648				# yuba
#BASE_LAT, BASE_LONG = 32.580, -117.090				# sanysidro
#BASE_LAT, BASE_LONG = 38.215, -122.662				# petaluma
BASE_LAT, BASE_LONG = 37.805, -122.295				# oakland

DELTA_LAT, DELTA_LONG = 0.02, 0.03
NUMBER_OF_STEPS = 8   	# will grab this many images squared times 2 (for 2010, 2012)
DIMENSION = 1000

# api info
#API_BASE_IMAGERY = 'http://waterapp.enviro-service.appspot.com/imagery/naip'
API_BASE_IMAGERY = 'http://genome.enviro-service.appspot.com/imagery/naip'

# read args
parser = argparse.ArgumentParser(description='Download NAIP imagery.')
parser.add_argument('prefix_name', type=str, help='Prefix for save image filename.')
parser.add_argument('save_location', type=str, help='Directory to save imagery.')
parser.add_argument('log_file', type=str, help='Filename to log images downloaded.')
args = parser.parse_args()

# make save directory if necessary
if not os.path.exists(args.save_location):
	os.makedirs(args.save_location)

# open log file
f_log = open(args.log_file,'a')

# build payload and make call for imagery
for i in range(NUMBER_OF_STEPS):
	mylat = BASE_LAT + i * DELTA_LAT
	for j in range(NUMBER_OF_STEPS):
		num = i * NUMBER_OF_STEPS + j
		mylong = BASE_LONG + j * DELTA_LONG
		payload = {}
		payload['lon'] = mylong
		payload['lat'] = mylat
		payload['dimension'] = DIMENSION
		payload['year'] = 2010			# do 2010 first
		payload['color'] = 'rgb'
		url_payload = urllib.urlencode(payload)
		full_url = API_BASE_IMAGERY + '?' + url_payload
		response = urllib2.urlopen(full_url)
		image = response.read()
		filename = "{}{:0>3d}-2010.png".format(args.prefix_name,num+1)	# pad num with zeros to 3 digits
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
		filename = "{}{:0>3d}-2012.png".format(args.prefix_name,num+1)	# pad num with zeros to 3 digits
		save_path = os.path.join(args.save_location,filename)
		with open(save_path,'wb') as f:
			f.write(image)
			log_string = "{0}, {1}\n".format(save_path,full_url)
			f_log.write(log_string)
			print 'Saved image to file: {0} from URL: {1}'.format(save_path,full_url)
			f.close()

f_log.close()
print 'Finished'

