# Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2508.02425-b31b1b.svg)](https://arxiv.org/abs/2508.02425)
[![IEEE CASE 2025](https://img.shields.io/badge/IEEE%20CASE-2025-blue.svg)]()

Official implementation of the paper "Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing" accepted at IEEE CASE 2025.

## 📖 Overview

This repository contains the implementation of a three-class human/object detection system for robot manipulators using proprioceptive sensing. The system enables robots to distinguish between humans, soft objects, and hard objects during physical human-robot collaboration (pHRC) by analyzing joint torque and position data.

**Key Highlights:**
- 🎯 **91.11% accuracy** achieved during real-time testing
- 🤖 Tested on **Franka Emika Panda** and **Universal Robots UR10** manipulators
- 📊 Three-class classification: Human, PVC, and Aluminum
- 🧠 Implements LSTM, GRU, and Transformer models
- 🔄 Real-time contact detection and classification

## 🌟 Features

- **Multi-Class Classification**: Distinguishes between human contact, soft objects (PVC), and hard objects (Aluminum)
- **Proprioceptive Sensing**: Uses only internal robot sensors (joint torque, position) - no external cameras required
- **Real-Time Operation**: Processes sensor data at 200 Hz for immediate contact detection
- **Multiple Robot Support**: Compatible with Franka Emika Panda and Universal Robots platforms
- **Data Collection Tools**: Includes scripts for recording new motions and collecting training data
- **Preprocessing Pipeline**: Comprehensive data preparation and augmentation strategies
- **Model Training**: Training scripts for LSTM, GRU, and Transformer architectures
- **Evaluation Tools**: Performance analysis and visualization utilities

## 🏗️ Repository Structure

```
HumanObjectDetection/
├── DataPreparation/        # Data preprocessing and augmentation
├── Evaluation/             # Model evaluation and analysis tools
├── ModelGeneration/        # Training scripts for ML models
├── _util/                  # Utility functions and helpers
├── dataLabeling/           # Data labeling and annotation tools
├── frankaRobot/           # Franka Emika Panda robot interface
│   ├── main.py            # Main robot control node
│   ├── saveDataNode.py    # Data collection node
│   └── recordNewMotion.py # Motion recording utility
├── urRobot/               # Universal Robots interface
│   └── main_ur10.py       # UR10 control script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- ROS Noetic
- CUDA-capable GPU (recommended)
- Franka Emika Panda robot OR Universal Robots UR10 (for hardware experiments)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MarH19/HumanObjectDetection.git
cd HumanObjectDetection
```

2. **Create conda environment**
```bash
conda create -n frankapyenv python=3.8
conda activate frankapyenv
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install robot-specific packages**

For Franka Emika Panda:
- Install [franka-interface](https://iamlab-cmu.github.io/franka-interface-docs/)
- Install [frankapy](https://iamlab-cmu.github.io/frankapy/)

For Universal Robots:
- Install [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/)

### Python Dependencies

```
numpy==1.19.5
torch==1.10.1
pandas==1.1.5
torchvision==0.11.2
torchmetrics==0.8.2
canlib
```

## 📊 Dataset

The dataset consists of real-time proprioceptive sensor data collected from robot manipulators during collisions with three object classes:

- **Human Class**: Collisions with dummy arm (Young's Modulus < 0.1 GPa)
- **PVC Class**: Collisions with rigid PVC tube (Young's Modulus ≈ 3.4 GPa)
- **Aluminum Class**: Collisions with aluminum profiles (Young's Modulus ≈ 68 GPa)

Data is recorded at 200 Hz sampling rate with multiple setup variations to ensure dataset diversity.

**Dataset Repository**: [MindLabZHAW/humanObjectDetectionDataset](https://github.com/MindLabZHAW/humanObjectDetectionDataset)

## 🤖 Usage

### Running with Franka Emika Panda

#### 1. Unlock the Robot
```
- Turn on the robot (wait for solid yellow light)
- Connect to robot desk (172.16.0.2 or 192.168.15.33)
- Unlock the robot (light should turn blue)
- Activate FCI
```

#### 2. Start FrankaPy Control
```bash
conda activate frankapyenv
bash /home/mindlab/franka/frankapy/bash_scripts/start_control_pc.sh -i localhost
```

#### 3. Launch Digital Glove Node (Terminal 2)
```bash
source /opt/ros/noetic/setup.bash
python /path/to/dataLabeling/sensorNode.py
```

#### 4. Start Robot Control Node (Terminal 3)
```bash
conda activate frankapyenv
source /opt/ros/noetic/setup.bash
source /path/to/franka-interface/catkin_ws/devel/setup.bash --extend
source /path/to/frankapy/catkin_ws/devel/setup.bash --extend
python3 frankaRobot/main.py
```

#### 5. Start Data Collection Node (Terminal 4)
```bash
source /opt/ros/noetic/setup.bash
source /path/to/franka-interface/catkin_ws/devel/setup.bash --extend
source /path/to/frankapy/catkin_ws/devel/setup.bash --extend
python3 frankaRobot/saveDataNode.py
```

### Running with Universal Robots UR10

#### 1. Activate Remote Control
Enable remote control from the robot teach pendant

#### 2. Run Control Program
```bash
conda activate frankapyenv
source /opt/ros/noetic/setup.bash
python3 urRobot/main_ur10.py
```

### Recording New Motions (Franka)

```bash
conda activate frankapyenv
source /opt/ros/noetic/setup.bash
source /path/to/franka-interface/catkin_ws/devel/setup.bash --extend
source /path/to/frankapy/catkin_ws/devel/setup.bash --extend
python3 frankaRobot/recordNewMotion.py
```

## 🧪 Model Training

1. **Prepare Data**
```bash
cd DataPreparation
python preprocess_data.py --input /path/to/raw/data --output /path/to/processed/data
```

2. **Train Models**
```bash
cd ModelGeneration
python train.py --model lstm --epochs 100 --batch_size 32
```

3. **Evaluate Performance**
```bash
cd Evaluation
python evaluate.py --model_path /path/to/model.pth --test_data /path/to/test/data
```

## 📈 Results

The best-performing model achieved **91.11% accuracy** during real-time testing on the Franka Emika Panda robot manipulator, demonstrating the feasibility of multi-class contact detection using proprioceptive sensing alone.

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{hehli2025multiclass,
  title={Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing},
  author={Hehli, Justin and Heiniger, Marco and Rezayati, Maryam and van de Venn, Hans Wernher},
  booktitle={2025 IEEE International Conference on Automation Science and Engineering (CASE)},
  year={2025},
  organization={IEEE}
}
```

ArXiv preprint:
```bibtex
@article{hehli2025multiclass,
  title={Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing},
  author={Hehli, Justin and Heiniger, Marco and Rezayati, Maryam and van de Venn, Hans Wernher},
  journal={arXiv preprint arXiv:2508.02425},
  year={2025}
}
```

## 👥 Authors

- **Justin Hehli**\* - University of Zurich & MINDLab, ZHAW - [justinsimon.hehli@uzh.ch](mailto:justinsimon.hehli@uzh.ch)
- **Marco Heiniger**\* - University of Zurich & MINDLab, ZHAW - [marcoh228@gmail.com](mailto:marcoh228@gmail.com)
- **Maryam Rezayati**† - MINDLab, ZHAW - [rzma@zhaw.ch](mailto:rzma@zhaw.ch)
- **Hans Wernher van de Venn**† - ZHAW - [vhns@zhaw.ch](mailto:vhns@zhaw.ch)

\* Equal contribution  
† Corresponding authors

## 🙏 Acknowledgments

This research was supported by the Eurostars project (Grant No. E!3087) titled **SmartSenseAI**.

## 🔗 Related Resources

- **Paper**: [arXiv:2508.02425](https://arxiv.org/abs/2508.02425)
- **Dataset**: [MindLabZHAW/humanObjectDetectionDataset](https://github.com/MindLabZHAW/humanObjectDetectionDataset)
- **Franka Interface**: [franka-interface documentation](https://iamlab-cmu.github.io/franka-interface-docs/)
- **FrankaPy**: [frankapy documentation](https://iamlab-cmu.github.io/frankapy/)
- **UR RTDE**: [ur_rtde documentation](https://sdurobotics.gitlab.io/ur_rtde/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues and Contributions

If you encounter any issues or have suggestions for improvements, please open an issue on GitHub. Contributions are welcome!
