# cell-measurement

![image](https://github.com/user-attachments/assets/71e68b5d-d26c-4464-a233-c913b9a396b2)

![image](https://github.com/user-attachments/assets/fde5d299-ab89-4540-a95e-8c1bb1074eed)


A collection of Python scripts for analyzing and measuring cell images using OpenCV.

## Requirements

- Python 3.x
- OpenCV (opencv-python)
- NumPy

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Scripts

### 1. Standard Measure (`standard_measure.py`)
- Basic cell measurement using grayscale threshold
- Shows width measurements of detected cells
- Uses binary thresholding

### 2. Gray Measure (`gray_measure.py`)
- Enhanced cell measurement using adaptive thresholding
- Better segmentation for varying lighting conditions
- Shows width measurements of detected cells

### 3. Gray Measure with Pixel Labels (`gray_mesuare_pixcel_label.py`)
- Color-based cell detection (red and green)
- Shows detailed measurements including:
  - Width and height in pixels
  - Area in square pixels

### 4. Gray Measure with Noise Reduction (`gray_mesuare_pixcel_label_reduce_noise.py`)
- Advanced version with noise filtering
- Only shows cells larger than 100x100 pixels
- Enlarged labels for better visibility

## Usage

1. Place your cell image as `cell_image.jpg` in the project directory
2. Run any of the scripts:
```bash
python standard_measure.py
# or
python gray_measure.py
# or
python gray_mesuare_pixcel_label.py
# or
python gray_mesuare_pixcel_label_reduce_noise.py
# or
python measure_single_particle.py
```

3. Press any key to close the image window

## Features
- Contour detection
- Cell measurements
- Color-based filtering
- Noise reduction
- Visual annotations
