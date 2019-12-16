#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 2019 4:57 pm

@author: doshininad
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense
from keras import losses, optimizers
from config import Config

class NetModel:
    def __init__(self, model_path):
        self.model = None
        self.x = None
        self.prediction = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()

        self._model()
        self.base_model = None

    def _model(self):
        input_shape = (Config.config['input_image_width'], 
                       Config.config['input_image_height'], 
                       Config.config['input_image_depth'])
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        self.x = self.base_model.output
        self.x = Dense(512, activation='relu')(self.x)
        self.x = Dense(256, activation='relu')(self.x)
        self.x = Dense(64, activation='relu')(self.x)
        self.prediction = Dense(Config.config['num_outputs'], activation='relu')(self.x)
        self.model = Model(inputs=self.base_model.input, outputs=self.prediction)
        
        for layer in self.base_model.layers[:44]:
            layer.trainable = False
        for layer in self.base_model.layers[44:]:
            layer.trainable = True
        
        self._compile()

    def _compile(self):
        self.model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam())

    def save(self):
        json_string = self.model.to_json()
        weight_filename = self.model_path + '_' + const.CONFIG_YAML
        open(weight_filename+'.json', 'w').write(json_string)
        self.model.save_weights(weight_filename+'.h5', overwrite=True)

    def load(self):
        from keras.models import model_from_json
        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()


