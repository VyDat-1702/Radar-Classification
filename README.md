# Radar-Communication Waveform Classification

Deep learning-based waveform classification for efficient spectrum management in 6G Cognitive Radio-IoT networks.

## Overview

As 6G networks integrate radar and communication systems, spectrum congestion becomes a critical challenge. This project addresses this issue by developing a lightweight deep learning model for accurate waveform classification on resource-constrained CR-IoT devices.

## Problem Statement

The convergence of radar and communications in 6G networks requires:
- Efficient spectrum utilization
- Real-time signal classification
- Low computational overhead for IoT devices
- Robust performance under realistic channel conditions

## Solution

We propose a cost-effective deep convolutional neural network (CNN) that:
- Learns essential radio features from time-frequency representations
- Incorporates attention mechanisms to reduce complexity
- Maintains high classification accuracy with minimal resources
- Optimized for edge computing deployment

## Key Features

- **Time-Frequency Representation (TFR)**: Converts signals into visual patterns for CNN processing
- **Lightweight Architecture**: Designed for resource-constrained devices
- **Attention Mechanisms**: Focuses on important features while reducing network size
- **Multi-Waveform Support**: Classifies 8 types of radar and communication signals
- **Edge-Ready**: Optimized for deployment on CR-IoT devices

## Dataset

- **Signal Types**: 8 radar and communication waveforms
- **Conditions**: Attenuated signals under realistic channel conditions
- **Input Format**: Time-frequency representations

## Model Architecture

```
Input: Time-Frequency Representation (TFR)
    ↓
Convolutional Layers (Lightweight)
    ↓
Attention Modules
    ↓
Classification Layer
    ↓
Output: Waveform Type (8 classes)
```

**Advantages**:
- Reduced computational complexity
- High classification accuracy
- Suitable for real-time processing
- Low memory footprint

## Requirements

```
python >= 3.8
tensorflow >= 2.8
numpy
matplotlib
scipy
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/radar-comm-classification.git
cd radar-comm-classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Train the model
python train.py --data_path ./dataset --epochs 100

# Evaluate on test set
python evaluate.py --model_path ./checkpoints/best_model.h5

# Classify a single signal
python classify.py --signal_path ./test_signal.npy
```

## Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/95371078-aac9-4167-a7ce-4f8eb302d759" alt="TFR Sample 1" width="500"/>
  <p><i>Time-Frequency Representation Example 1</i></p>
  <br/>
  
  <img src="https://github.com/user-attachments/assets/c080b990-39a7-4787-be43-9865007a1c76" alt="TFR Sample 2" width="500"/>
  <p><i>Time-Frequency Representation Example 2</i></p>
  <br/>
  
  <img src="https://github.com/user-attachments/assets/973392a2-54e2-4ee3-9029-5413f049199c" alt="Model Performance" width="500"/>
  <p><i>Classification Performance Metrics</i></p>
</div>

## Performance Metrics

Our model demonstrates:
- High classification accuracy across all 8 waveform types
- Robust performance under channel attenuation
- Efficient inference time suitable for real-time applications
- Significant complexity reduction compared to baseline models

## Applications

- **Cognitive Radio Networks**: Dynamic spectrum access
- **6G Communications**: Efficient resource allocation
- **IoT Devices**: Low-power signal classification
- **Spectrum Monitoring**: Automated signal identification
- **Interference Management**: Real-time waveform detection

## Future Work

- Extend to more waveform types
- Test on real-world hardware
- Optimize for mobile deployment
- Integrate with CR-IoT protocols
- Add adversarial robustness

## Acknowledgments

This research contributes to the advancement of 6G cognitive radio technologies and efficient spectrum management for IoT devices.

## Contact

For questions or collaboration, please open an issue in this repository.

---

**Note**: This is a research project focused on efficient waveform classification for next-generation wireless networks.
