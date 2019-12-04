#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 11 14:35 2019

@author: ninad
"""
import const
from keras.models import Sequential, Model
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, Concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input
from keras import losses, optimizers

from config import Config

class NetModel:
    def __init__(self, model_path):
        self.model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        self.config = Config()

        self._model()
        
    ###########################################################################
    #
    def _model(self):
        
        if self.config.net_model_type == const.NET_TYPE_CE491:

            input_shape = (const.IMAGE_HEIGHT, const.IMAGE_WIDTH, const.IMAGE_DEPTH)
            
            self.model = Sequential([
                          Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
                          Conv2D(24, (5, 5), strides=(2,2), activation='relu'),
                          Conv2D(36, (5, 5), strides=(2,2), activation='relu'),
                          Conv2D(48, (5, 5), strides=(2,2), activation='relu'),
                          Conv2D(64, (3, 3), activation='relu'),
                          Conv2D(64, (3, 3), activation='relu'),
                          Flatten(),
                          Dense(100, activation='relu'),
                          Dense(50, activation='relu'),
                          Dense(10, activation='relu'),
                          Dense(self.config.num_outputs)])
            self.model.summary()
            self._compile()

        ###----------------------------------------------------------------###
        else:
            print('Select proper type of Model')
            return


        
        
    ##########################################################################
    #
    def _compile(self):
        self.model.compile(loss=losses.mean_squared_error,
                  optimizer=optimizers.Adam())


    ###########################################################################
    #
    # save model
    def save(self):
        
        json_string = self.model.to_json()
        weight_filename = self.model_path+'_n'+str(self.config.net_model_type)
        open(weight_filename+'.json', 'w').write(json_string)
        self.model.save_weights(weight_filename+'.h5', overwrite=True)
    
    
    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def load(self):
        
        from keras.models import model_from_json
        
        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()
        
    ###########################################################################
    #
    # show summary
    def summary(self):
        self.model.summary()
        
