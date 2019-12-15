#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
#

import const

class Config:
    def __init__(self): # model_name):
        self.version = (const.VERSION_MAJOR, const.VERSION_MINOR) 
        self.valid_rate = const.VALID_RATE
        self.fname_ext = const.IMAGE_EXT
        self.data_ext = const.DATA_EXT
        
        self.data_shuffle = const.DATA_SHUFFLE
        self.num_epochs = const.NUM_EPOCH
        self.batch_size = const.BATCH_SIZE
        self.num_outputs = const.NUM_OUTPUT   # steering_angle, throttle
        
        self.raw_scale = const.RAW_SCALE      # Multiply raw input by this scale
        self.jitter_tolerance = const.JITTER_TOLERANCE # joystick jitter
       
        self.net_model_type = const.NET_TYPE_JAEROCK
        self.aug_flip = const.AUG_FLIP
        self.aug_bright = const.AUG_BRIGHT
        self.aug_shift = const.AUG_SHIFT

        self.image_size = (const.IMAGE_WIDTH, const.IMAGE_HEIGHT, const.IMAGE_DEPTH)
        self.capture_area = (const.CROP_X1, const.CROP_Y1, const.CROP_X2, const.CROP_Y2)
        
    def summary(self):
        print('========== config ==========')
        print('- version ----------')
        print(self.version)
        print('- training ----------')
        print('-- valid_rate: ' + str(self.valid_rate))
        print('-- data_shuffle: ' + str(self.data_shuffle))
        print('-- num_epochs: ' + str(self.num_epochs))
        print('-- batch_size: ' + str(self.batch_size))
        print('-- num_outputs: ' + str(self.num_outputs))
        print('-- raw_scale: ' + str(self.raw_scale))
        print('-- jitter_tolerance: ' + str(self.jitter_tolerance))
        print('-- net_model_type: ' + str(self.net_model_type))
        print('-- aug_flip: ' + str(self.aug_flip))
        print('-- aug_bright: ' + str(self.aug_bright))
        print('-- aug_shift: ' + str(self.aug_shift))
        print('-- image_size: ' + str(self.image_size[0]) + 'x' + str(self.image_size[1]))
        print('- data collection ------------')
        print('-- capture_area : ')
        print(self.capture_area)
        