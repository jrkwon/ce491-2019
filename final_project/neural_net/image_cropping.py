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
import shutil
from config import Config

###############################################################################
#       
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith(extension))

###############################################################################
#       
def crop():
    config = Config()
    path = sys.argv[1]
    dirs = list_files(path, config.fname_ext)

    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            fname, ext = os.path.splitext(fullpath)
            cropped = im.crop((config.capture_area[0], config.capture_area[1],
                              config.capture_area[2], config.capture_area[3])) #corrected
            cropped.save(fname + '_crop' + ext)
            print('Cropped - ' + fname + ext)


###############################################################################
#       
def main():
    try:
        if (len(sys.argv) != 2):
            print('Give a folder name of drive data.')
            return
        
        crop()

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       

###############################################################################
#       
if __name__ == '__main__':
    main()