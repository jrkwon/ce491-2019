#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:15:46 2019

@author: jaerock
"""

import sys

from drive_log import DriveLog

###############################################################################
#       
def log(weight_name, data_folder_name):
    drive_log = DriveLog(weight_name, data_folder_name) 
    drive_log.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python log.py weight_name data_folder_name')
        
        log(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
