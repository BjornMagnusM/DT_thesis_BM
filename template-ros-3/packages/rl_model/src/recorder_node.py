#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class Recorder:
    def __init__(self):

        self.out = cv2.VideoWriter(
            "duckie.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (640, 480)
        )

        rospy.Subscriber(
            "/duckiebot14/camera_node/image/compressed",
            CompressedImage,
            self.cb,
            queue_size=1
        )

    def cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return

        img = cv2.resize(img, (640, 480))
        self.out.write(img)

if __name__ == "__main__":
    rospy.init_node("recorder")
    Recorder()
    rospy.spin()