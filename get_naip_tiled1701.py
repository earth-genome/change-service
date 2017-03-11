#get_naip_tiled1701.py
"""Quickly mashed (01/07/17) version of get_naip_list.py/get_naip_tiled.py
for quick full tiling (or almost full -- my DELTA_LAT and DELTA_LON are approximations) of a 500 image swath 10 images x 50 images (roughly 20km x 100km) starting at Ocean Beach and stepping east.
"""

import sys
import os
import urllib
import urllib2
import json
import numpy as np
import pdb


# api info
API_BASE_IMAGERY = 'http://genome.enviro-service.appspot.com/imagery/naip'
COLOR = 'rgb'
DIMENSION = 1000
YEARS = [2010,2012]

def build_latlon(coords):
    """Build a list of latitudes/longitudes from base coords."""
    latlon_list = []
    for step_NS in range(coords['steps'][0]):
        for step_EW in range(coords['steps'][1]):
            latlon = (coords['base'] +
                np.array([step_NS*coords['delta'][0],
                          step_EW*coords['delta'][1]]))
            latlon_list.append(latlon)
    print 'Built lat/lon coordinates: {}'.format(latlon_list)
    return latlon_list
    
def image_pull(fileprefix,latlon):
    """Pull images from coordinate list latlon and write to file."""

    # create save directory, logfile
    savedir = fileprefix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    f_log = open(os.path.join(savedir,fileprefix+'.log'),'a')
    

    payload = {}
    payload['color'] = COLOR
    payload['dimension'] = DIMENSION
    
    # build payload and make call for imagery
    images_to_clean = os.listdir(savedir)
    images_to_clean.remove(fileprefix+'.log')
    ct = 0
    for (mylat,mylon) in latlon:
        payload['lon'] = mylon
        payload['lat'] = mylat
        for y in YEARS:
            filename = (str(mylat)+str(mylon)+
                        '-dim'+str(DIMENSION)+
                        '-'+str(y)+'.png')
            save_path = os.path.join(savedir,filename)
            if not os.path.exists(save_path):
                payload['year'] = y	
                url_payload = urllib.urlencode(payload)
                full_url = API_BASE_IMAGERY + '?' + url_payload
                response = urllib2.urlopen(full_url)
                image = response.read()

                with open(save_path,'wb') as f:
                    f.write(image)
                f_log.write("{0}, {1}\n".format(save_path,full_url))
                print 'Saved: {0} from URL: {1}'.format(save_path,full_url)
                ct += 1
            else:
                images_to_clean.remove(filename)
    print "Downloaded {} new image files.".format(ct)
    print "Removing {} old image files.".format(len(images_to_clean))
    for img_file in images_to_clean:
        os.remove(os.path.join(savedir,img_file))
        print 'Removed {}'.format(img_file)
    f_log.close()

if __name__ == '__main__':
    #coords = {'name': 'sutrobaths',
    #          'base': np.array([37.780,-122.505]),
    #          'delta': np.array([-.018,.023]),
    #          'steps': [10,50]}
    coords = {'name': 'lajolla',
              'base': np.array([32.75,-117.25]),
              'delta': np.array([.018,.023]),
              'steps': [30,16]}
    latlon = build_latlon(coords)
    image_pull(coords['name'],latlon)
