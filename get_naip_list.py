#get_naip_list.py
"""
Get NAIP imagery.
Argument: filename containing  lat/lon coordinates of images to be pulled:
  Format: filename-latlon.txt. Comma separated values lat,lon,lat,lon...
  with comments ('#').
"""
import sys
import os
import urllib
import urllib2
import json
import numpy as np
import pdb

# api info
#API_BASE_IMAGERY = 'http://waterapp.enviro-service.appspot.com/imagery/naip'
API_BASE_IMAGERY = 'http://genome.enviro-service.appspot.com/imagery/naip'
COLOR = 'rgb'
DIMENSION = 1000
YEARS = [2010,2012]


def read_latlon(infile):
    """Parse infile for multiple lat/lon coordinates; return as list."""
    latlon = []
    for line in infile:
        digitlist = line.split('#')[0].split(',')
        for lat,lon in zip(digitlist[1::2],digitlist[::2]):
            latlon.append([float(lat),float(lon)])
    print 'Parsed lat/lon coordinates: {}'.format(latlon)
    return latlon
            

def image_pull(fileprefix,latlon):
    """Pull images from coordinate list latlon and write to file."""

    # make save, log directories
    savedir = fileprefix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    f_log = open(os.path.join(savedir,fileprefix+'.log'),'a')

    payload = {}
    payload['color'] = COLOR
    payload['dimension'] = DIMENSION
    
    # build payload and make call for imagery
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
    print "Downloaded {} new image files.".format(ct)
    f_log.close()

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
        inf = open(filename,'r')
    except (IndexError, IOError):
        sys.exit("Syntax: $ get_naip_list.py filename-latlon.txt")
    prefix = filename.split('-latlon.txt')[0]
    latlon = read_latlon(inf)
    image_pull(prefix,latlon)
    
    
