#!/usr/bin/env python3
import numpy as np
import torch
import cv2
from cv_bridge import CvBridge


import rospy
from sensor_msgs.msg import CompressedImage

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
)

# Helper file with encoder and agent 
from sac import Actor, CropResizeWrapperROS, ImgWrapperROS,FrameStackObservationROS

class AgentNode(DTROS):
    def __init__(self, node_name):
        super(AgentNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        #Initilize the agent 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(obs_shape=(12, 84, 84), action_dim=2).to(self.device)


        self.model_path = rospy.get_param("~model_path")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()



        self.CropResizeWrapperROS = CropResizeWrapperROS(shape=(84, 84))
        self.ImgWrapperROS = ImgWrapperROS()
        self.FrameStackObservationROS = FrameStackObservationROS(stack_size=4)

        #Initilize bridge for de compressing 
        self.bridge = CvBridge()

        self.latest_obs = None

        # Publisher this is changed from "~car_cmd" to "/duckiebot14/car_cmd_switch_node/cmd" 
        #               where original was from core and new based on topics on acual duckiebot
        self.car_cmd = rospy.Publisher(
            "/duckiebot14/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # Subscribers
        self.sub_image = rospy.Subscriber(
            "/duckiebot14/camera_node/image/compressed", CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )

        # Update Parameters timer
        rospy.Timer(rospy.Duration.from_sec(0.1), self.control_loop) ##Should decrease timer to publish more often 

    def image_cb(self, image_msg):
        """
        Processes the incoming image messages.
        """
        #rospy.loginfo("Start of image_cb")

        # Decode from compressed image with OpenCV
        obtained_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
    
        #Resize image to same dim as sim 
        obtained_image =  self.CropResizeWrapperROS.observation(obtained_image)

        # Prepare observation: (C, H, W) -> (1, C, H, W)
        obtained_image = self.ImgWrapperROS.observation(obtained_image)

        # Frame stack set to 4 like in training  
        #if len(self.FrameStackObservationROS.obs_queue) == 0:
        if self.latest_obs is None:
            stacked = self.FrameStackObservationROS.reset(obtained_image)
        else:
            stacked = self.FrameStackObservationROS.append(obtained_image)

        self.latest_obs = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(self.device)

    def control_loop(self, event):

        #rospy.loginfo("Start of control_loop")

        #Return if there is no observation 
        if self.latest_obs is None:
            return

        with torch.no_grad():
            # Use mean_action for deterministic evaluation
            _, _, action = self.actor.get_action(self.latest_obs)

        action = action.cpu().numpy().reshape(-1)

        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()
        car_control_msg.v = float(action[0])
        car_control_msg.omega = float(action[1])

        self.car_cmd.publish(car_control_msg)
        rospy.loginfo(f"Published car_cmd: v={car_control_msg.v}, omega={car_control_msg.omega}")

    def onShutdown(self):
        rospy.loginfo("[AgentNode] Shutdown. Stopping robot...")
    
        ## added for stopping the robot 
        stop_msg = Twist2DStamped()
        stop_msg.header.stamp = rospy.Time.now()
        stop_msg.v = 0.0
        stop_msg.omega = 0.0

        for _ in range(3):
            self.car_cmd.publish(stop_msg)
            rospy.sleep(0.05)
    



if __name__ == "__main__":
    agent_node = AgentNode(node_name="agent_node")
    rospy.on_shutdown(agent_node.onShutdown)
    rospy.spin()