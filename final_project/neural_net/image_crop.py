# -*- coding: utf-8 -*-
"""
Created on Wed Dec 3 13:55:21 2019

@author: Jaerock Kwon
"""

#####
# We don't need this cropping process anymore
#####

from PIL import Image
import os, sys
import const
from config import Config

###############################################################################
#       
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith(extension))

###############################################################################
#       
def main(data_path):
    #config = Config()
    dirs = list_files(data_path, const.IMAGE_EXT)

    for item in dirs:
        fullpath = os.path.join(data_path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            fname, ext = os.path.splitext(fullpath)
            cropped = im.crop((Config.config['image_crop_x1'], Config.config['image_crop_y1'],
                              Config.config['image_crop_x2'], Config.config['image_crop_y2'])) 
            cropped.save(fname + '_crop' + ext)
            print('Cropped - ' + fname + ext)



###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ image_crop data_path')
        
        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       
