#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2
import numpy as np
#import keras
import sklearn

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
from data_augmentation import DataAugmentation

###############################################################################
#
class DriveTrain:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56/'
    def __init__(self, data_path):
        
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path
        csv_path = data_path + '/' + model_name + const.DATA_EXT  # use it for csv file name 
        
        self.csv_path = csv_path
        self.train_generator = None
        self.valid_generator = None
        self.train_hist = None
        self.drive = None
        
        #self.config = Config() #model_name)
        
        self.data_path = data_path
        #self.model_name = model_name
        
        self.drive = DriveData(self.csv_path)
        self.net_model = NetModel(data_path)
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()
        
        
    ###########################################################################
    #
    def _prepare_data(self):
    
        self.drive.read()
        
        from sklearn.model_selection import train_test_split
        
        samples = list(zip(self.drive.image_names, self.drive.measurements))
        self.train_data, self.valid_data = train_test_split(samples, 
                                   test_size=Config.config['validation_rate'])
        
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
                                          
    ###########################################################################
    #
    def _build_model(self, show_summary=True):

        def _generator(samples, batch_size=Config.config['batch_size']):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
               
                if Config.config['lstm'] is False:
                    samples = sklearn.utils.shuffle(samples)

                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset+batch_size]
        
                    images = []
                    measurements = []

                    for image_name, measurement in batch_samples:
                        
                        image_path = self.data_path + '/' + image_name
                        image = cv2.imread(image_path)
                        image = cv2.resize(image, 
                                           (Config.config['input_image_width'],
                                            Config.config['input_image_height']))
                        image = self.image_process.process(image)
                        images.append(image)
        
                        steering_angle, throttle = measurement
                        
                        if abs(steering_angle) < Config.config['steering_angle_jitter_tolerance']:
                            steering_angle = 0
                        
                        measurements.append(steering_angle*Config.config['steering_angle_scale'])
                        
                        if Config.config['data_aug_flip'] is True:    
                            # Flipping the image
                            flip_image, flip_steering = self.data_aug.flipping(image, steering_angle)
                            images.append(flip_image)
                            measurements.append(flip_steering*Config.config['steering_angle_scale'])
    
                        if Config.config['data_aug_bright'] is True:    
                            # Changing the brightness of image
                            if steering_angle > Config.config['steering_angle_jitter_tolerance'] or \
                                steering_angle < -Config.config['steering_angle_jitter_tolerance']:
                                bright_image = self.data_aug.brightness(image)
                                images.append(bright_image)
                                measurements.append(steering_angle*Config.config['steering_angle_scale'])
    
                        if Config.config['data_aug_shift'] is True:    
                            # Shifting the image
                            shift_image, shift_steering = self.data_aug.shift(image, steering_angle)
                            images.append(shift_image)
                            measurements.append(shift_steering*Config.config['steering_angle_scale'])

                    X_train = np.array(images)
                    y_train = np.array(measurements)

                    if Config.config['lstm'] is True:
                        X_train = np.array(images).reshape(-1, 1, 
                                          Config.config['input_image_height'],
                                          Config.config['input_image_width'],
                                          Config.config['input_image_depth'])
                        y_train = np.array(measurements).reshape(-1, 1, 1)
                    
                    if Config.config['lstm'] is False:
                        yield sklearn.utils.shuffle(X_train, y_train)
                    else:
                        yield X_train, y_train
        
        self.train_generator = _generator(self.train_data)
        self.valid_generator = _generator(self.valid_data)
        
        if (show_summary):
            self.net_model.model.summary()
    
    ###########################################################################
    #
    def _start_training(self):
        
        if (self.train_generator == None):
            raise NameError('Generators are not ready.')
        
        ######################################################################
        # callbacks
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        
        # checkpoint
        callbacks = []
        #weight_filename = self.net_model.name + '_' + const.CONFIG_YAML + '_ckpt'
        weight_filename = self.data_path + '_' + const.CONFIG_YAML + '_ckpt'
        checkpoint = ModelCheckpoint(weight_filename+'.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)
        
        self.train_hist = self.net_model.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//Config.config['batch_size'], 
                epochs=Config.config['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//Config.config['batch_size'],
                verbose=1, callbacks=callbacks)
    

    ###########################################################################
    #
    def _plot_training_history(self):
    
        print(self.train_hist.history.keys())
        
        ### plot the training and validation loss for each epoch
        plt.plot(self.train_hist.history['loss'])
        plt.plot(self.train_hist.history['val_loss'])
        plt.ylabel('mse loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validatation set'], loc='upper right')
        plt.show()
        
        
    ###########################################################################
    #
    def train(self, show_summary=True):
        
        self._prepare_data()
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save()
        self._plot_training_history()
        Config.summary()
