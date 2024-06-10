
# Neuromorphic Keyword Spotting with Pulse Density Modulation MEMS Microphones

## Overview
This repository contains the source code implementation for the paper titled "Neuromorphic Keyword Spotting with Pulse Density Modulation MEMS Microphones". In this paper, we propose a novel approach that directly connects Pulse Density Modulation (PDM) microphones to Spiking Neural Networks (SNNs). This direct connection eliminates intermediate stages, significantly reducing computational costs while achieving high accuracy in Keyword Spotting tasks.

## Code Structure
- **gsc_dataset.py**: This module contains functionalities related to Google Speech Commands dataset loading, PDM encoding, and data augmentation.
- **paralif.py**: The Parallelizable Leaky Integrate-and-Fire (LIF) neuron code is implemented in this module.
- **paralif_model.py**: This module contains the network architecture and the code for implementing axonal delays.
- **run_train.py**: The training and evaluation code is provided in this module.
- **run_job.py**: Configuration parameters for running the training and evaluation processes are specified in this module.

## Getting Started
To utilize the code provided in this repository, follow these steps:
1. Clone the repository to your local machine.
2. Ensure all required dependencies are installed. Refer to the `requirements.txt` file for details.
3. Execute `run_job.py` to commence training and evaluation of the proposed PDM-to-SNN connection system.

## Citation
If you find this work useful and utilize the code or ideas presented here, please consider citing our paper:

\[Coming...\]

## Contact
For any inquiries or assistance regarding the code or paper, feel free to contact the authors:
- [Arnaud](mailto:sidi.yaya.arnaud.yarga@usherbrooke.ca)