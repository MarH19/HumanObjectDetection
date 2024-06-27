#!/bin/bash

# Function to execute the pipeline
pipeline() {
    bash script1.sh
    run_frankapy
    run_sensor_node
    run_robot_node
    run_save_data_node
}

# Function to run frankapy
run_frankapy() {
    echo "Starting frankapy..."
    gnome-terminal -- bash -c 'conda activate frankapyenv; bash /home/mindlab/franka/frankapy/bash_scripts/start_control_pc.sh -i localhost; exec bash'
}

# Function to run sensor node
run_sensor_node() {
    echo "Starting sensor node..."
    read -p "Do you want to execute the sensor node? (y/n): " sensor_choice
    if [ "$sensor_choice" == "y" ] || [ "$sensor_choice" == "Y" ]; then
        gnome-terminal -- bash -c 'source /opt/ros/noetic/setup.bash; /home/mindlab/miniconda3/envs/frankapyenv/bin/python /home/mindlab/humanObjectDetection/dataLabeling/sensorNode.py; exec bash'
    else
        echo "Skipping execution of sensor node."
    fi
}

# Function to run robot node
run_robot_node() {
    echo "Starting robot node..."
    read -p "Do you want to execute the robot node? (y/n): " robot_choice
    if [ "$robot_choice" == "y" ] || [ "$robot_choice" == "Y" ]; then
        gnome-terminal -- bash -c 'conda activate frankapyenv; source /opt/ros/noetic/setup.bash; source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend; source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend; /home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/main.py; exec bash'
    else
        echo "Skipping execution of robot node."
    fi
}

# Function to run save data node
run_save_data_node() {
    echo "Starting save data node..."
    read -p "Do you want to execute the save data node? (y/n): " save_choice
    if [ "$save_choice" == "y" ] || [ "$save_choice" == "Y" ]; then
        gnome-terminal -- bash -c 'source /opt/ros/noetic/setup.bash; source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend; source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend; /home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/saveDataNode.py; exec bash'
    else
        echo "Skipping execution of save data node."
    fi
}

# Display the pipeline
echo "Pipeline: 
1. bash script1.sh
2. run frankapy
3. run sensor node
4. run robot node
5. run save data node
"

# Ask for confirmation to execute the rest of the pipeline
read -p "Do you want to execute the rest of the pipeline? (y/n): " pipeline_choice

case "$pipeline_choice" in
  y|Y ) 
    echo "Starting execution of the rest of the pipeline..."
    pipeline
    echo "Pipeline execution completed."
    ;;
  n|N ) 
    echo "Pipeline execution cancelled."
    ;;
  * ) 
    echo "Invalid choice. Exiting without executing the rest of the pipeline."
    ;;
esac
