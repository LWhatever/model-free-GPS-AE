# Model-Free End-to-End Deep Learning of Joint Geometric and Probabilistic Shaping for Optical Fiber Communication in IM/DD System

## Overview

This Jupyter Notebook is designed to implement and train a model-free autoencoder that facilitates the concurrent optimization of both geometric and probabilistic constellation shaping in communication systems. The notebook includes various steps such as pre-training, loading channel parameters, defining transceiver models, and training an equalizer. Below is a brief description of the key files and their roles in this project.

## Files Description

1.**g_cos_55.txt & g_sin_55.txt**

    - These files contain the pulse shaping pair for the CAP (Carrierless Amplitude Phase) modulation.

2.**W_t_btb.txt**

    - This file stores a set of coefficients for an FIR (Finite Impulse Response) filter. The FIR filter simulates a real ISI-limited channel.

3.**helper_lite.py**

    - This Python script contains a set of necessary functions required by the Jupyter Notebook. These functions include various utility functions for signal processing and performance evaluation, which are used in the implementation and training of the joint PS and GS autoencoder model.

4.**A0_TransInit.m & A0_TransInit.py**

    - First, Run the .m file to generate standard bit-to-constellation mapping pairs

    - Then, Run the .py file to pre-train the bit-to-constellation GS mapper.

    - The pre-trained GS mappers are loaded in the Notebook.

## Workflow

The workflow in the Jupyter Notebook includes the following steps:

1.**Initialization and Parameter Setting**

    - Import necessary libraries and set initial parameters for the communication system and autoencoder model.

2.**Function Definitions**

    - Define various functions required for signal processing, modulation, and performance evaluation.

3.**Pre-training**

    - Pre-train a one-hot to bits mapper to initialize the training process.

4.**Channel Loading**

    - Load the FIR channel coefficients from`W_t_btb.txt` and visualize the channel response.

5.**Transceiver Model Definition**

    - Define the transceiver models, including the logit encoder, one-hot encoder, and constellation mapper.

6.**Equalizer Training**

    - Pre-Train an equalizer with a matched filter to mitigate the initial effects of ISI and enabel the autoencoder to have a good initial state before start training.

7.**Model Training**

    - Train the autoencoder model using the defined functions and parameters, and evaluate its performance.

## Requirement

- Tensorflow, version: 2.3.0
- Tensorflow probability, version: 0.11.1
- matplotlib, version: 3.4.3
- numpy, version: 1.18.5
- scipy, version: 1.4.1

## Acknowledgment

This project extensively leverages the functionalities of the following relevant projects:

[Rassibassi/claude: End-to-end learning of optical communication systems](https://github.com/Rassibassi/claude)
