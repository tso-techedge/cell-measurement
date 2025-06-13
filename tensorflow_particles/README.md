# Particle Detection with TensorFlow

This project implements a machine learning solution for detecting and measuring particles in microscopy images using TensorFlow.

## Project Structure

- `tensorflow_particles.py` - Main script for particle detection
- `tensorflow_training.py` - Model training script
- `particle_detector.keras` - Trained model
- `training/` - Directory containing training data
- `particle_0.png` - Sample image for testing

## Requirements

```txt
tensorflow
opencv-python
numpy
scikit-learn
```

## How It Works

### Training Data Generation

The system uses synthetic particle data generated with controlled parameters:
- Image size: 224x224 pixels
- Particle radius range: 25-50 pixels
- Includes random noise for robustness

### Model Architecture

The CNN model consists of:
- Input shape: (224, 224, 1)
- 3 Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers
- Output: 3 values (x, y, radius)

### Particle Detection

The detection pipeline:
1. Loads and preprocesses grayscale images
2. Runs inference using trained model
3. Converts predictions to image coordinates
4. Validates detected particles
5. Visualizes results with measurements

## Usage

1. Train the model:
```sh
python machine_learning/tensorflow_training.py
```

2. Run detection:
```sh
python machine_learning/tensorflow_particles.py
```

The script will:
- Load a test image
- Detect particles
- Display and save results with measurements
- Output labeled image as 'labeled_particles.png'

## Output

The detection results show:
- Detected particles outlined in green
- Particle measurements including:
  - Diameter (px)
  - Area (px²)
- Labels for up to 10 randomly selected particles
