# Contact Interpretation System

numpy==1.19.5

torch==1.10.1

pandas==1.1.5

torchvision==0.11.2

torchmetrics==0.8.2

## robot APIS: 

### Franka Emika Panda

franka-interface: Welcome to franka-interface’s Documentation! — franka-interface 1.0.0 documentation (iamlab-cmu.github.io)
![image](https://github.com/MindLabZHAW/contactInterpretation/assets/10871265/d7c654f5-4e37-4f47-b552-87ba16e3cf78)

frankapy: Welcome to FrankaPy’s Documentation! — frankapy 1.0.0 documentation (iamlab-cmu.github.io)
![image](https://github.com/MindLabZHAW/contactInterpretation/assets/10871265/fd57b7fb-077c-4a1a-a05a-048f2ba25365)

### Universal Robots
Universal Robots RTDE: https://sdurobotics.gitlab.io/ur_rtde/

## Run robot and contact detection

### Franka Robot:

#### 1st  Step: unlock robot
	1) connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	2) unlock the robot and activate FCI

#### 2nd Step: run frankapy

open an terminal

	conda activate frankapyenv
	cd /home/mindlab/franka
	bash run_frankapy.sh

#### 3nd Step: run digital glove node

open another temrinal

	source /opt/ros/noetic/setup.bash
	/home/mindlab/miniconda3/envs/frankapyenv/bin/python /home/mindlab/contactInterpretation/dataLabeling/digitalGloveNode.py

#### 4th Step: run robot node

open another terminal 

	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
	source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend

	/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/contactInterpretation/frankRobot/main.py

#### 5th Step: run save data node

open another terminal

	source /opt/ros/noetic/setup.bash
	source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
	source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend

	/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/contactInterpretation/frankaRobot/saveDataNode.py



### to chage publish rate of frankastate go to : 
sudo nano /home/mindlab/franka/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch


### UR Robot

#### 1st  Step: activate remote control from robot teach pandanent


#### 2nd Step: run program

open a terminal
conda activate frankapyenv
source /opt/ros/noetic/setup.bash
/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/urRobot/main_ur10.py


## record new motion for franka robot:
#### 1st  Step: unlock robot
	1) connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	2) unlock the robot and activate FCI

#### 2nd Step: run frankapy

open an terminal

	conda activate frankapyenv
	cd /home/mindlab/franka
	bash run_frankapy.sh

#### 3nd Step: run this code

open another terminal 

	conda activate frankapyenv
	source /opt/ros/noetic/setup.bash
	source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
	source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend

	/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/contactInterpretation/frankRobot/recordNewMotion.py
