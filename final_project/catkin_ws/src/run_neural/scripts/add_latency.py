#!/usr/bin/env python

import sys
import os

import rospy
from sensor_msgs.msg import Image

sys.path.append('../neural_net/')
os.chdir('../neural_net/')

from config import Config


def get_fps(latency_ms):
    return round(1000.0 / latency_ms)


class Handler(object):

    def __init__(self):
        self.msg = Image()
        self.picked_msg = Image()

    def callback(self, img_msg):
        self.msg = img_msg

    def pick_msg(self):
        self.picked_msg = self.msg


if __name__ == '__main__':
    h = Handler()

    rospy.init_node('add_latency')
    rospy.Subscriber('/bolt/front_camera/image_raw', Image, h.callback, queue_size=1)
    pub = rospy.Publisher('/delayed_img', Image, queue_size=10)

    conf = Config().config
    latency_ms = conf['lc_latency_ms']
    fps = get_fps(latency_ms)
    print('Latency: {} ms ({} Hz)'.format(latency_ms, fps))
    rate = rospy.Rate(fps)

    while not rospy.is_shutdown():
        # Publish the message picked in the last loop before the sleeping
        pub.publish(h.picked_msg)

        # Pick new image and sleep, hence, simulating the processing delay
        h.pick_msg()
        rate.sleep()
