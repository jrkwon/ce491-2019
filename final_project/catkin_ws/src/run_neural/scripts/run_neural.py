#!/usr/bin/env python

import cv2
import time
import rospy
import numpy as np
from bolt_msgs.msg import Control
from sensor_msgs.msg import Image

import sys
import os

sys.path.append('../neural_net/')

from image_converter import ImageConverter
from config import Config

import tensorflow as tf
from keras import losses

conf = Config().config

# Model path
model_path = conf['lc_model']
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
# init_steps = conf['lc_init_steps']
# accl_steps = conf['lc_accl_steps']
# neut_steps = conf['lc_neut_steps']
# curve_threshold = conf['lc_curve_threshold']
# BRAKE_APPLY_SEC = conf['lc_brake_apply_sec']
# THROTTLE_DEFAULT = conf['lc_ throttle_default']
# THROTTLE_SHARP_TURN = conf['lc_throttle_sharp_turn']

# Calculate final crop
crop_y1 = gazebo_crop_y1 + crop_y_neural_net
crop_y2 = gazebo_crop_y2
crop_x1 = gazebo_crop_x1
crop_x2 = rightside_width_cut


class SimpleThrottle(object):

    magnitude = 0.2
    buffer_size = 400
    buffer = None

    def __init__(self):
        print('Using simple throttle with value {}'.format(self.magnitude))
        self.buffer = [self.magnitude] * self.buffer_size

    def get_throttle(self, steer):
        try:
            val = self.buffer.pop()
        except IndexError:
            val = 0.0

        return val, False, 0.0


class Throttle(object):
    init_steps = 100
    curve_threshold = 0.4   # 0.115
    straight_accl_steps = 80
    straight_neut_steps = 40
    curved_accl_steps = 80
    curved_neut_steps = 40
    brake_steps = 70
    straight_throttle = 0.3
    curved_throttle = 0.2
    neut_throttle = 0.0
    soft_brake_val = 0.35

    def __init__(self):
        self.straight_buffer = [self.straight_throttle] * self.init_steps
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
        self.curved_buffer = []
        try:
            return self.straight_buffer.pop(), self.brake, self.soft_brake_val
        except IndexError:
            self._refill_straight_buffer()
            return self.straight_buffer.pop(), self.brake, self.soft_brake_val

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

    def get_curved_throttle(self, steer):
        self.straight_buffer = []
        try:
            return self.curved_buffer.pop(), self.brake, self.soft_brake_val
        except IndexError:
            self._refill_curved_buffer()
            return self.curved_buffer.pop(), self.brake, self.soft_brake_val

    def get_throttle(self, steer):
        if abs(steer) >= self.curve_threshold:
            return self.get_curved_throttle(steer)
        else:
            return self.get_straight_throttle()


class Model(object):

    def __init__(self, model_path):
        self.model_path = model_path
        self.config_path = os.path.join(self.model_path, 'config.json')
        self.weight_path = os.path.join(self.model_path, 'weights.h5')
        self.params_path = os.path.join(self.model_path, 'params.json')
        self.model = self._load()

    def print_params(self):
        with open(self.params_path) as f:
            params = f.read()

        print('=== Model parameters ===')
        print(params)
        print('========================')

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
        self.print_params()
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

    def __init__(self, model_path, latency_mode, throttle_mode):
        self.model = Model(model_path)
        self.throttle = SimpleThrottle() if throttle_mode == 'simple' else Throttle()
        self.processor = Preprocessor()

        rospy.init_node('run_neural')
        if latency_mode == 'no_latency':
            print('No latency mode.')
            rospy.Subscriber('/bolt/front_camera/image_raw', Image, self._callback, queue_size=1)
        elif latency_mode == 'latency':
            print('Latency mode.')
            rospy.Subscriber('/delayed_img', Image, self._callback, queue_size=1)

        self.publisher = rospy.Publisher('/bolt', Control, queue_size=10)
        self.rate = rospy.Rate(ros_rate)

    def _callback(self, imgmsg):
        self.processor.process(imgmsg)

    def publish(self, data):
        self.publisher.publish(data)


def main(latency_mode='no_latency'):
    # Initialize manager with trained model
    manager = Manager(model_path, latency_mode, throttle_mode='adaptive')
    
    joy_data = Control()
    steer = 0.0  # Initial steering value

    t0 = time.time()
    while not rospy.is_shutdown():

        # If processor is ready with image, update steer value
        if manager.processor.is_ready:
            steer = manager.model.predict(manager.processor.image)

        # Get throttle value
        throttle, is_brake, brake_val = manager.throttle.get_throttle(steer)

        # Publish joy_data
        joy_data.steer = steer
        if is_brake:
            joy_data.brake = brake_val
            joy_data.throttle = 0.0
        else:
            joy_data.throttle = throttle
            joy_data.brake = 0.0
        manager.publish(joy_data)

        time_elapsed = round(time.time() - t0, 1)
        print('Throttle: {} | Brake {} | Time Elapsed: {} | Steer: {}'.format(
            format(round(joy_data.throttle, 2), '.2f'),
            format(round(joy_data.brake, 2), '.2f'),
            time_elapsed,
            format(round(np.squeeze(joy_data.steer), 3), '.3f')
        ))
        manager.rate.sleep()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
