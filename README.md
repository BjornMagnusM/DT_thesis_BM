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
Go in to template-ros-3 

    cd template-ros-3
Building the code 

    docker build -t dt-ros:latest .
Building the bash file 

    chmod +x run_docker.sh
Running a bash script to connect to the docker given the name of the duckiebot

    ./run_docker.sh duckiebot14
Launch the node with the file name of the model, the model need to be in models folder for the package, runs the final model

    roslaunch rl_model rl_model.launch model_name:=OvalSAC1MnoDis
See the ros topics 
   
    rostopic list





















    






