#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os

class Recorder:
    def __init__(self):
        duckie_name = os.getenv("DUCKIE_NAME", "duckie")
        rospy.loginfo(f"the duckiebot's name is {duckie_name}")
        #Video writer 
        self.out = cv2.VideoWriter("/data/duckie.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480)
        )
        # Subscribers
        self.sub_image = rospy.Subscriber(
            f"/{duckie_name}/camera_node/image/compressed", CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )


    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.out.write(img)

if __name__ == "__main__":
    rospy.init_node("recorder")
    Recorder()
    rospy.spin()