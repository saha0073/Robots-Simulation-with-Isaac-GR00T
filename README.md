# GR00T Integration with IsaacSim Robots

This repository demonstrates the integration of NVIDIA's GR00T (General Robot Operator & Orchestrator Transformer), a foundational model for robotics, with various robots in the IsaacSim environment. GR00T represents a significant advancement in robotics AI as it's trained directly from video data, enabling it to learn general robotic behaviors across different morphologies and tasks without task-specific training.

## Overview

This project explores its application with several IsaacSim robots:

- **Dexterous Hands** (`hands_gr00t.py`): Implementation with Allegro and Shadow hands
- **Franka & Carter** (`franka_carter_gr00t.py`): Integration with Franka robotic arm and Carter mobile platform
- **H1 Locomotion** (`h1_gr00t_locomotion.py`): Bipedal robot locomotion control

## Key Features

- Direct integration of GR00T with IsaacSim's physics simulation
- Real-time control and visualization in the IsaacSim environment


## Requirements

- IsaacSim/IsaacLab environment
- NVIDIA GR00T model weights
- Python 3.10
- PyTorch
- Required IsaacSim/Lab packages

## Future Work

The next phase of development will focus on:
1. Fine-tuning GR00T with custom datasets for specific use cases
2. Improving stability and performance for each robot type

