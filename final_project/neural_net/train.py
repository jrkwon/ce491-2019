#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from drive_train import DriveTrain

###############################################################################
#
def main(data_folder_name):
    drive_train = DriveTrain(data_folder_name)
    drive_train.train(show_summary=False)


###############################################################################
#
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python train.py data_path')

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
