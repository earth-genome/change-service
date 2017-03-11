#get_naip_imagery.py
"""
Get NAIP imagery.
Requires two arguments: a directory to save imagery, and a filename for the log.
"""

import argparse
import numpy as np
import urllib
import urllib2
import json
import os

# params
YEAR1 = 2010
YEAR2 = 2012
COUNTY_INDEX = 25
NUMBER_OF_LOCATIONS_PER_COUNTY = 8
# for 1m res
#URL_PREFIX = 'waterapp'
#DIM = 256
# for dim1000
#URL_PREFIX = 'genome'
#DIM = 1000
# for dim500
URL_PREFIX = 'genome'
DIM = 500

# api info
API_BASE_COORDS = 'http://genome.enviro-service.appspot.com/land/validation'
API_BASE_IMAGERY = ('http://'+URL_PREFIX+
    '.enviro-service.appspot.com/imagery/naip')

landuse_types = ['cropland-natural vegetation mosaic','urban and built-up','evergreen needleleaf forest','closed shrublands','deciduous broadleaf forest','grasslands','snow and ice','mixed forest','water','open shrublands','savannas','croplands','permanent wetlands','barren or sparsely vegetated','woody savannas','deciduous needleleaf forest','evergreen broadleaf forest']

county_names = ['Tulare','Calaveras','Merced','San Luis Obispo','Riverside','Los Angeles','Orange','San Bernardino','Sonoma','Marin','Humboldt','Mono','Del Norte','Modoc','Solano','Ventura','Santa Cruz','Yuba','Placer','Glenn','Trinity','Mendocino','Yolo','Imperial','Stanislaus','Colusa','Alameda','El Dorado','Sutter','Inyo','San Benito','Monterey','Kings','Sierra','Lassen','Lake','San Diego','Mariposa','Nevada','Tehama','San Francisco','Alpine','Madera','Sacramento','Santa Barbara','Plumas','Santa Clara','San Mateo','Butte','San Joaquin','Tuolumne','Napa','Siskiyou','Kern','Contra Costa','Fresno','Amador','Shasta']


# read args
parser = argparse.ArgumentParser(description='Download NAIP imagery.')
parser.add_argument('save_location', type=str, help='Location to save imagery.')
parser.add_argument('log_file', type=str, help='Filename to log images downloaded.')
args = parser.parse_args()

# build payload and make call for random lat/long service
payload = {}
payload['n'] = NUMBER_OF_LOCATIONS_PER_COUNTY
payload['landuse_type'] = landuse_types[1]
payload['county'] = county_names[COUNTY_INDEX]
url_payload = urllib.urlencode(payload)
full_url = API_BASE_COORDS + '?' + url_payload
print "Getting random coordinates for {0} county, from URL: {1}".format(county_names[COUNTY_INDEX],full_url)

# receive response and decode
response = urllib2.urlopen(full_url)
data = json.load(response)
print "Received coordinates:"
for coord in data['results']:
	print coord

# build payload and make call for imagery

# make county name pretty for file saving
county_name = ''.join(county_names[COUNTY_INDEX].split()).lower()

# make save directory if necessary
if not os.path.exists(args.save_location):
	os.makedirs(args.save_location)

# open log file
f_log = open(args.log_file,'a')

for n,coords in enumerate(data['results']):
	mylong,mylat = coords
	payload = {}
	payload['lon'] = mylong
	payload['lat'] = mylat
	payload['dimension'] = DIM
	payload['year'] = YEAR1			# do 2010 first
	url_payload = urllib.urlencode(payload)
	full_url = API_BASE_IMAGERY + '?' + url_payload
	response = urllib2.urlopen(full_url)
	image = response.read()
	filename = "{}{:0>3d}-{}.jpg".format(county_name,n+1,YEAR1)	# pad n+1 with zeros to 3 digits
	save_path = os.path.join(args.save_location,filename)
	with open(save_path,'wb') as f:
		f.write(image)
		log_string = "{0}, {1}\n".format(save_path,full_url)
		f_log.write(log_string)
		print 'Saved image to file: {0} from URL: {1}'.format(save_path,full_url)
		f.close()
	payload['year'] = YEAR2		# now do 2012
	url_payload = urllib.urlencode(payload)
	full_url = API_BASE_IMAGERY + '?' + url_payload
	response = urllib2.urlopen(full_url)
	image = response.read()
	filename = "{}{:0>3d}-{}.jpg".format(county_name,n+1,YEAR2)
	save_path = os.path.join(args.save_location,filename)
	with open(save_path,'wb') as f:
		f.write(image)
		log_string = "{0}, {1}\n".format(save_path,full_url)
		f_log.write(log_string)
		print 'Saved image to file: {0} from URL: {1}'.format(save_path,full_url)
		f.close()

f_log.close()
print 'Finished'

