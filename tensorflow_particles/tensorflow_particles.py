import cv2
import numpy as np
import tensorflow as tf
import random


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input.

    Args:
        image: Grayscale image array
        target_size: Tuple of (height, width)
    Returns:
        Preprocessed image with shape (1, height, width, 1)
    """
    # Resize image
    image = cv2.resize(image, target_size)
    # Convert to float and normalize
    image = image.astype("float32") / 255.0
    # Add channel and batch dimensions
    image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)


# Load the trained model
model = tf.keras.models.load_model("particle_detector.keras")

# Load and preprocess image
image = cv2.imread("particle_0.png", cv2.IMREAD_GRAYSCALE)  # Force grayscale
if image is None:
    raise ValueError("Could not load image particles_image.png")

print(f"Original image shape: {image.shape}")
original_size = image.shape

# Preprocess image - simplified pipeline
processed_image = cv2.resize(image, (224, 224))
processed_image = processed_image.astype("float32") / 255.0
processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension
processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

print(f"Processed image shape: {processed_image.shape}")

# Detect particles using TensorFlow
predictions = model.predict(processed_image)

print(f"Raw predictions shape: {predictions.shape}")
print(f"Raw predictions values: {predictions}")

# Handle predictions based on shape
predictions = predictions.reshape(-1, 3)  # Ensure shape is (N, 3)
print(f"Reshaped predictions: {predictions}")

# Convert predictions to original image coordinates
valid_circles = []
for pred in predictions:
    x = int(pred[0] * original_size[1])
    y = int(pred[1] * original_size[0])
    # Use absolute value for radius since negative values don't make sense
    radius = int(abs(pred[2]) * min(original_size))
    print(f"Converted coordinates: x={x}, y={y}, radius={radius}")

    # Validate coordinates are within image bounds
    if (
        0 <= x < original_size[1] and 0 <= y < original_size[0] and 10 <= radius <= 100
    ):  # Relaxed radius constraints for testing
        valid_circles.append((x, y, radius))
        print(f"Added valid circle: x={x}, y={y}, radius={radius}")

print(f"Number of valid circles found: {len(valid_circles)}")

# Convert to color image for visualization
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Ensure we have circles to process
if not valid_circles:
    print("No valid particles detected!")
    cv2.putText(
        image_color,
        "No valid particles detected!",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
else:
    # Randomly select up to 10 particles and label them
    selected_circles = random.sample(valid_circles, min(10, len(valid_circles)))

    # Draw selected circles and labels with improved visibility
    for i, (x, y, radius) in enumerate(selected_circles):
        # Draw circle
        cv2.circle(image_color, (x, y), radius, (0, 255, 0), 2)

        # Calculate measurements
        diameter = radius * 2
        area = np.pi * radius * radius

        # Draw label background
        label = f"#{i+1} D={diameter:.0f}px A={area:.0f}px²"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Position label above the circle
        text_x = max(0, x - w // 2)
        text_y = max(20, y - radius - 10)

        # Draw white background for text
        cv2.rectangle(
            image_color,
            (text_x - 2, text_y - h - 2),
            (text_x + w + 2, text_y + 2),
            (255, 255, 255),
            -1,
        )

        # Draw text
        cv2.putText(
            image_color,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

# Save the result
cv2.imwrite("labeled_particles.png", image_color)

# Show result
cv2.imshow("Particle Detection Results", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
