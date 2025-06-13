import cv2
import numpy as np

# Load the image
image = cv2.imread("particles_image.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding for segmentation
thresholded = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Find contours of particles
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out noise (only particles larger than 50x50 pixels)
filtered_contours = [
    c for c in contours if cv2.boundingRect(c)[2] > 50 and cv2.boundingRect(c)[3] > 50
]

# Draw contours and label size
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
    radius = int((w + h) / 4)  # Approximate radius for circular particles
    area = cv2.contourArea(contour)  # Calculate area
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw contour

    # Adjust label placement
    text_x = max(10, x)
    text_y = max(20, y - 10)

    label_text = f"R={radius}px, {area}px²"
    cv2.putText(
        image, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
    )

# Show the result
cv2.imshow("Filtered Particles with Labels", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
