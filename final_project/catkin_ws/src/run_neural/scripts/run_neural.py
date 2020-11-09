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
from keras import losses, optimizers

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
SHARP_TURN_MIN = conf['lc_sharp_turn_min_speed']
BRAKE_APPLY_SEC = conf['lc_brake_apply_sec']
THROTTLE_DEFAULT = conf['lc_ throttle_default']
THROTTLE_SHARP_TURN = conf['lc_throttle_sharp_turn']

# Calculate final crop
crop_y1 = gazebo_crop_y1 + crop_y_neural_net
crop_y2 = gazebo_crop_y2
crop_x1 = gazebo_crop_x1
crop_x2 = rightside_width_cut
# class NeuralControl:
#     def __init__(self, weight_file_name):
#         rospy.init_node('run_neural')
#         self.ic = ImageConverter()
#         self.image_process = ImageProcess()
#         self.rate = rospy.Rate(30)
#         self.drive= DriveRun(weight_file_name)
#         rospy.Subscriber('/bolt/front_camera/image_raw', Image, self._controller_cb)
#         self.image = None
#         self.image_processed = False
#         #self.config = Config()
#         self.braking = False
#
#     def _controller_cb(self, image):
#         img = self.ic.imgmsg_to_opencv(image)
#         cropped = img[Config.config['image_crop_y1']:Config.config['image_crop_y2'],
#                       Config.config['image_crop_x1']:Config.config['image_crop_x2']]
#
#         img = cv2.resize(cropped, (Config.config['input_image_width'],
#                                    Config.config['input_image_height']))
#
#         self.image = self.image_process.process(img)
#
#         ## this is for CNN-LSTM net models
#         if Config.config['lstm'] is True:
#             self.image = np.array(self.image).reshape(1,
#                                  Config.config['input_image_height'],
#                                  Config.config['input_image_width'],
#                                  Config.config['input_image_depth'])
#         self.image_processed = True
#
#     def timer_cb(self):
#         self.braking = False


def preprocess_opened_image(img):
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    img = cv2.resize(img, (fin_width, fin_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


class NeuralControlLatency:
    def __init__(self, config_path, weight_path):
        self.config_path = config_path
        self.weight_path = weight_path
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.rate = rospy.Rate(ros_rate)
        self.model = self.load_model()
        rospy.Subscriber('/bolt/front_camera/image_raw', Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        self.braking = False
        self.throttle_steps = 200
        self.throttle_buffer = [THROTTLE_DEFAULT] * 400
        self.last_status_accelerating = True

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
        self.image_processed = True

    def predict(self, preprocessed_img):
        return self.model.predict(preprocessed_img, batch_size=1)

    def stop_brake_cb(self):
        self.braking = False

    def get_accelerating_throttle_buffer(self):
        return [THROTTLE_DEFAULT] * self.throttle_steps

    def get_neutral_throttle_buffer(self):
        return [0.0] * self.throttle_steps

    def refill_throttle_buffer(self):
        if not self.last_status_accelerating:
            self.throttle_buffer = self.get_accelerating_throttle_buffer()
            self.last_status_accelerating = True
        else:
            self.throttle_buffer = self.get_neutral_throttle_buffer()
            self.last_status_accelerating = False


def main():
    # Initialize trained model
    neural_control = NeuralControlLatency(CONFIG_PATH, WEIGHT_PATH)
    
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher('/bolt', Control, queue_size=10)
    joy_data = Control()

    print('\nStart running. Vroom. Vroom. Vroooooom......')

    t0 = time.time()
    while not rospy.is_shutdown():
        if neural_control.image_processed is False:
            continue
        
        # predicted steering angle from an input image
        prediction = neural_control.predict(neural_control.image)
        joy_data.steer = prediction

        # Throttle strategy 1
        # if neural_control.braking is False:
        #     if abs(joy_data.steer) > SHARP_TURN_MIN:
        #         joy_data.throttle = THROTTLE_SHARP_TURN
        #         joy_data.brake = 0.5
        #         neural_control.braking = True
        #         timer = threading.Timer(BRAKE_APPLY_SEC, neural_control.stop_brake_cb)
        #         timer.start()
        #     else:
        #         joy_data.throttle = THROTTLE_DEFAULT
        #         joy_data.brake = 0
        # else:
        #     joy_data.throttle = THROTTLE_SHARP_TURN
        #     joy_data.brake = 0.5

        # Throttle strategy 2
        try:
            throttle = neural_control.throttle_buffer.pop()
        except IndexError:
            neural_control.refill_throttle_buffer()
            throttle = neural_control.throttle_buffer.pop()

        joy_data.throttle = throttle

        print('Steer: %f | Throttle: %f | Brake: %f' % (joy_data.steer, joy_data.throttle, joy_data.brake))

        # publish joy_data
        joy_pub.publish(joy_data)
        # t1 = time.time()
        # print('Process time: {}'.format(round(t1 - t0, 5)))
        # t0 = t1

        # ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')