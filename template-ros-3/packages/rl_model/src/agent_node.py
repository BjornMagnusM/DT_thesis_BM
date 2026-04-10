#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
)

class AgentNode(DTROS):
    def __init__(self, node_name):
        super(AgentNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        self.v = 0.0
        self.omega = -0.2

        # Publisher this is changed from "~car_cmd" to "/duckiebot14/car_cmd_switch_node/cmd" 
        #               where original was from core and new based on topics on acual duckiebot
        self.car_cmd = rospy.Publisher(
            "/duckiebot14/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # Update Parameters timer
        rospy.Timer(rospy.Duration.from_sec(1.0), self.pub_cmd) ##Should decrease timer to publish more often 

    def pub_cmd(self, event):
        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()

        car_control_msg.v = self.v
        car_control_msg.omega = self.omega
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