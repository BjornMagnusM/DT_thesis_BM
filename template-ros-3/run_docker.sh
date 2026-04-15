#!/bin/bash
DUCKIE_NAME=$1
DUCKIE_IP=$(getent hosts duckiebot14.local | awk '{print $1}')

docker run -it --rm \
  --network host \
  --add-host duckiebot14.local:$DUCKIE_IP \
  dt-ros:latest bash