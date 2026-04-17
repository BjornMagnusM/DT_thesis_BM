#!/bin/bash
DUCKIE_NAME=$1
DUCKIE_IP=$(getent hosts ${DUCKIE_NAME}.local | awk '{print $1}')
MY_IP=$(hostname -I | awk '{print $1}')

docker run -it --rm \
  --network host \
  --add-host ${DUCKIE_NAME}.local:$DUCKIE_IP \
  -e DUCKIE_NAME=${DUCKIE_NAME} \
  -e ROS_IP=$MY_IP \
  -e ROS_MASTER_URI=http://${DUCKIE_IP}:11311 \
  dt-ros:latest bash