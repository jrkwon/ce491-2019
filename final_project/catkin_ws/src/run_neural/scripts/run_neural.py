#!/usr/bin/env python

import threading 
import cv2
import time
import rospy
import numpy as np
from bolt_msgs.msg import Control
from sensor_msgs.msg import Image

import sys
import os

sys.path.append('../neural_net/')
os.chdir('../neural_net/')

from image_converter import ImageConverter
from config import Config

import tensorflow as tf
from keras import losses

conf = Config().config

# Model path
CONFIG_PATH = conf['lc_model_config']
WEIGHT_PATH = conf['lc_model_weight']
init_height = conf['lc_init_img_height']
init_width = conf['lc_init_img_width']
fin_height = conf['lc_fin_img_height']
fin_width = conf['lc_fin_img_width']
crop_y_neural_net = conf['lc_crop_y_start']
rightside_width_cut = conf['lc_rightside_width_cut']
ros_rate = conf['lc_ros_rate']
gazebo_crop_x1 = conf['image_crop_x1']
gazebo_crop_y1 = conf['image_crop_y1']
gazebo_crop_x2 = conf['image_crop_x2']
gazebo_crop_y2 = conf['image_crop_y2']

# Throttle params
init_steps = conf['lc_init_steps']
accl_steps = conf['lc_accl_steps']
neut_steps = conf['lc_neut_steps']
SHARP_TURN_MIN = conf['lc_sharp_turn_min_speed']
BRAKE_APPLY_SEC = conf['lc_brake_apply_sec']
THROTTLE_DEFAULT = conf['lc_ throttle_default']
THROTTLE_SHARP_TURN = conf['lc_throttle_sharp_turn']

# Calculate final crop
crop_y1 = gazebo_crop_y1 + crop_y_neural_net
crop_y2 = gazebo_crop_y2
crop_x1 = gazebo_crop_x1
crop_x2 = rightside_width_cut


class Throttle(object):

    def __init__(self):
        self.buffer = [THROTTLE_DEFAULT] * init_steps
        self.last_status_accl = False

    @staticmethod
    def _get_accl_throttle_buffer():
        return [THROTTLE_DEFAULT] * accl_steps

    @staticmethod
    def _get_neut_throttle_buffer():
        return [0.0] * neut_steps

    def _refill_buffer(self):
        if not self.last_status_accl:
            self.buffer = self._get_accl_throttle_buffer()
            self.last_status_accl = True
        else:
            self.buffer = self._get_neut_throttle_buffer()
            self.last_status_accl = False

    def get_throttle(self):
        try:
            return self.buffer.pop()
        except IndexError:
            self._refill_buffer()
            return self.buffer.pop()


class Model(object):

    def __init__(self, config_path, weight_path):
        self.config_path = config_path
        self.weight_path = weight_path
        self.model = self._load()

    def _load(self):
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

    def predict(self, preprocessed_img):
        return self.model.predict(preprocessed_img, batch_size=1)


class Preprocessor(object):

    def __init__(self):
        self.ic = ImageConverter()
        self.is_ready = False
        self._image = None

    @staticmethod
    def _preprocess_opened_image(img):
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        img = cv2.resize(img, (fin_width, fin_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img) / 255.0
        return np.expand_dims(img, axis=0)

    @property
    def image(self):
        self.is_ready = False
        return self._image

    @image.setter
    def image(self, val):
        self._image = val
        self.is_ready = True

    def process(self, img_msg):
        img = self.ic.imgmsg_to_opencv(img_msg)
        self.image = self._preprocess_opened_image(img)


class Manager(object):

    def __init__(self, config_path, weight_path, mode):
        self.model = Model(config_path, weight_path)
        self.throttle = Throttle()
        self.processor = Preprocessor()

        rospy.init_node('run_neural')
        if mode == 'no_latency':
            print('No latency mode.')
            rospy.Subscriber('/bolt/front_camera/image_raw', Image, self._callback, queue_size=1)
        elif mode == 'latency':
            print('Latency mode.')
            rospy.Subscriber('/delayed_img', Image, self._callback, queue_size=1)

        self.publisher = rospy.Publisher('/bolt', Control, queue_size=10)
        self.rate = rospy.Rate(ros_rate)

    def _callback(self, imgmsg):
        self.processor.process(imgmsg)

    def publish(self, data):
        self.publisher.publish(data)


def main(mode='no_latency'):
    # Initialize manager with trained model
    manager = Manager(CONFIG_PATH, WEIGHT_PATH, mode)
    
    joy_data = Control()
    steer = 0.0  # Initial steering value

    while not rospy.is_shutdown():

        # If processor is ready with image, update steer value
        if manager.processor.is_ready:
            steer = manager.model.predict(manager.processor.image)

        # Get throttle value
        throttle = manager.throttle.get_throttle()

        # Publish joy_data
        joy_data.steer = steer
        joy_data.throttle = throttle
        manager.publish(joy_data)

        print('Steer: {:03f} | Throttle: {:02f}'.format(np.squeeze(joy_data.steer),
                                                        joy_data.throttle))
        manager.rate.sleep()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
