#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017

@author: jaerock
"""


import sys

from drive_test import DriveTest
    

###############################################################################
#       
def main(trained_model_name, data_path):
    drive_test = DriveTest(trained_model_name)
    drive_test.test(data_path)    
       

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python test.py model_name data_path')

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
