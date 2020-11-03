#!/usr/bin/env python

import threading 
import cv2
import time
import rospy
import numpy as np
from bolt_msgs.msg import Control
from std_msgs.msg import Int32
from sensor_msgs.msg import Image

import sys
import os
import imageio

sys.path.append('../neural_net/')
# print(os.getcwd())
os.chdir('../neural_net/')

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

import tensorflow as tf
from keras import losses, optimizers

sys.path.append('../../../latcom/')
os.chdir('../../../latcom/')

# from preprocess import preprocess_opened_image


# Model path
CONFIG_PATH = '/home/sanjyot/bimi/robotics/av/working/latcom/models/model1_config_only.json'
WEIGHT_PATH = '/home/sanjyot/bimi/robotics/av/working/latcom/models/model1_weights_only.h5'

# TODO: Put these variables in config
height, width = (210, 800)
final_shape = (40, 80)
crop_y = 30
rightside_width_cut = 700

final_width, final_height = final_shape
crop_window = tf.constant([crop_y, 0, height - crop_y, rightside_width_cut], dtype=tf.int32)


def preprocess_opened_image(img):
    img = img[380:590, 0:800]
    img = img[crop_y:, :rightside_width_cut]
    img = cv2.resize(img, (final_height, final_width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


SHARP_TURN_MIN = 0.3
BRAKE_APPLY_SEC = 1.5
THROTTLE_DEFAULT = 0.2
THROTTLE_SHARP_TURN = 0.05

class NeuralControl:
    def __init__(self, weight_file_name):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(weight_file_name)
        rospy.Subscriber('/bolt/front_camera/image_raw', Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False

    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[Config.config['image_crop_y1']:Config.config['image_crop_y2'],
                      Config.config['image_crop_x1']:Config.config['image_crop_x2']]
                      
        img = cv2.resize(cropped, (Config.config['input_image_width'],
                                   Config.config['input_image_height']))
                                  
        self.image = self.image_process.process(img)

        ## this is for CNN-LSTM net models
        if Config.config['lstm'] is True:
            self.image = np.array(self.image).reshape(1, 
                                 Config.config['input_image_height'],
                                 Config.config['input_image_width'],
                                 Config.config['input_image_depth'])
        self.image_processed = True
        
    def timer_cb(self):
        self.braking = False


class NeuralControlLatency:
    def __init__(self, config_path, weight_path):
        self.config_path = config_path
        self.weight_path = weight_path
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.rate = rospy.Rate(30)
        self.model = self.load_model()
        self.config = Config()
        rospy.Subscriber('/bolt/front_camera/image_raw', Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        self.braking = False

    def load_model(self):
        with open(self.config_path) as json_file:
            json_config = json_file.read()
        model = tf.keras.models.model_from_json(json_config)

        model.load_weights(self.weight_path)
        model.compile(
            loss=losses.mean_squared_error,
            optimizer='adam',
            metrics=['mse']
        )
        return model

    def _controller_cb(self, imgmsg):
        img = self.ic.imgmsg_to_opencv(imgmsg)
        self.image = preprocess_opened_image(img)
        # print(self.image.shape)
        imageio.imwrite('sample_img.jpg', self.image[0])
        self.image_processed = True

    def predict(self, preprocessed_img):
        return self.model.predict(preprocessed_img, batch_size=1)

        
def main(weight_file_name):

    # ready for neural network
    neural_control = NeuralControlLatency(CONFIG_PATH, WEIGHT_PATH)
    
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher('/bolt', Control, queue_size = 10)
    joy_data = Control()

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt: \tbrake')

    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        # predicted steering angle from an input image
        if isinstance(neural_control, NeuralControl):
            prediction = neural_control.drive.run(neural_control.image)
        elif isinstance(neural_control, NeuralControlLatency):
            prediction = neural_control.predict(neural_control.image)
        joy_data.steer = prediction

        print('Steer: ', prediction)
        # print(joy_data)

        #############################
        ## TODO: you need to change the vehicle speed wisely  
        ## e.g. not too fast in a curved road and not too slow in a straight road
        
        # if brake is not already applied and sharp turn
        if neural_control.braking is False: 
            if abs(joy_data.steer) > SHARP_TURN_MIN: 
                joy_data.throttle = THROTTLE_SHARP_TURN
                joy_data.brake = 0.5
                neural_control.braking = True
                timer = threading.Timer(BRAKE_APPLY_SEC, neural_control.timer_cb) 
                timer.start()
            else:
                joy_data.throttle = THROTTLE_DEFAULT
                joy_data.brake = 0
        
            
        ## publish joy_data
        joy_pub.publish(joy_data)

        ## print out
        if Config.config['lstm'] is True:
            cur_output = '{0:.3f} \t{1:.2f} \t{2:.2f}\r'.format(prediction[0][0][0], 
                          joy_data.throttle, joy_data.brake)
        else:
            cur_output = '{0:.3f} \t{1:.2f} \t{2:.2f}\r'.format(prediction[0][0], 
                          joy_data.throttle, joy_data.brake)

        sys.stdout.write(cur_output)
        sys.stdout.flush()
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')