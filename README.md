# Duckietown Thesis 



[Duckietown](http://duckietown.com/) self-driving car simulator environments for OpenAI Gym.


## General dts command 
See available duckiebots 

    dts fleet discover
Pinging duckiebot 

    ping duckiebot14.local 
Shutdown duckie bot 

    dts duckiebot shutdown  duckiebot14
Open duckietown dashboard 

    dts duckiebot dashboard  duckiebot14
Standard password for duckiebots 

    quackquack
Controlling duckiebot with keyboard

    dts duckiebot keyboard_control duckiebot14



## Running simulator inside /dt-sim
Run training sac 

    python rl_bm/sac_continous_actions.py
Run eval

    python rl_bm/eval_sac.py     --model-path runs/Oval_sac_90k_Nodis/models/sac_step_Final.cleanrl_model     --num-episodes 5     --render
RUn visualize (doesn't work for me)

    python rl_bm/visualize_sac.py     --checkpoint runs/Oval_sac_90k_Nodis/models/sac_step_Final.cleanrl_model


## Running duckiebot 
Building the code 

    buildx build   --build-arg NCPUS=18   --build-arg ARCH=amd64   --file Dockerfile   --tag dt-template-ros-test:latest .
Connect via docker 

    docker run -it --rm --network host dt-template-ros-test:latest bash
Exports

    export VEHICLE_NAME=duckiebot14
    export ROS_MASTER_URI=http://192.168.1.7:11311
    export ROS_IP=192.168.1.6
Launch the node 

    roslaunch bm_test bm_test.launch
See the ros topics 
   
    rostopic list





















    






