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
curve_threshold = conf['lc_curve_threshold']
BRAKE_APPLY_SEC = conf['lc_brake_apply_sec']
THROTTLE_DEFAULT = conf['lc_ throttle_default']
THROTTLE_SHARP_TURN = conf['lc_throttle_sharp_turn']

# Calculate final crop
crop_y1 = gazebo_crop_y1 + crop_y_neural_net
crop_y2 = gazebo_crop_y2
crop_x1 = gazebo_crop_x1
crop_x2 = rightside_width_cut


class Throttle(object):
    straight_accl_steps = 130
    straight_neut_steps = 150
    curved_accl_steps = 130
    curved_neut_steps = 150
    brake_steps = 40
    straight_throttle = 0.2
    curved_throttle = 0.2
    neut_throttle = 0.0

    def __init__(self):
        self.straight_buffer = [self.straight_throttle] * init_steps
        self.curved_buffer = []

        self.straight_status_accl = False
        self.curved_status_accl = False
        self.last_status_straight = False
        self.brake = False

    @staticmethod
    def _get_buffer(throttle, steps):
        return [throttle] * steps

    def _refill_straight_buffer(self):
        if not self.straight_status_accl:
            self.straight_buffer = self._get_buffer(self.straight_throttle, self.straight_accl_steps)
            self.straight_status_accl = True
        else:
            self.straight_buffer = self._get_buffer(self.neut_throttle, self.curved_accl_steps)
            self.straight_status_accl = False

    def get_straight_throttle(self):
        self.last_status_straight = True
        self.brake = False
        try:
            return self.straight_buffer.pop(), self.brake
        except IndexError:
            self._refill_straight_buffer()
            return self.straight_buffer.pop(), self.brake

    def _refill_curved_buffer(self):
        if not self.brake and self.last_status_straight:
            self.curved_buffer = [0.0] * self.brake_steps
            self.brake = True
            return

        if not self.curved_status_accl:
            self.curved_buffer = self._get_buffer(self.curved_throttle, self.curved_accl_steps)
            self.curved_status_accl = True
            self.brake = False
            self.last_status_straight = False
        else:
            self.curved_buffer = self._get_buffer(self.neut_throttle, self.curved_accl_steps)
            self.curved_status_accl = False
            self.brake = False
            self.last_status_straight = False

    def get_curved_throttle(self):
        try:
            return self.curved_buffer.pop(), self.brake
        except IndexError:
            self._refill_curved_buffer()
            return self.curved_buffer.pop(), self.brake

    def get_throttle(self, steer):
        if abs(steer) >= curve_threshold:
            return self.get_curved_throttle()
        else:
            return self.get_straight_throttle()


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
        throttle, is_brake = manager.throttle.get_throttle(steer)

        # Publish joy_data
        joy_data.steer = steer
        if is_brake:
            joy_data.brake = 0.75
        else:
            joy_data.throttle = throttle
            joy_data.brake = 0.0
        manager.publish(joy_data)

        print('Throttle: {} | Brake {} | Steer: {}'.format(format(round(joy_data.throttle, 2), '.2f'),
                                                           format(round(joy_data.brake, 2), '.1f'),
                                                           format(round(np.squeeze(joy_data.steer), 3), '.3f')))
        manager.rate.sleep()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
