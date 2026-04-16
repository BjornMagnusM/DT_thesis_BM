#!/bin/bash
DUCKIE_NAME=$1
DUCKIE_IP=$(getent hosts ${DUCKIE_NAME}.local | awk '{print $1}')

docker run -it --rm \
  --network host \
  --add-host ${DUCKIE_NAME}.local:$DUCKIE_IP \
  -v ~/Downloads:/videos \
  dt-ros:latest bash