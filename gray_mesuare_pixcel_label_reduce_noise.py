import cv2
import numpy as np

# Load the image
image = cv2.imread("cell_image.jpg")

# Convert to HSV for better color detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for red and green
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Create masks for red and green colors
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Combine both masks
merged_mask = cv2.bitwise_or(mask_red, mask_green)

# Find contours of merged cells
contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out noise and draw contours with larger labels
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box

    # Ignore small noise (width or height less than 10px)
    if w > 100 and h > 100:
        area = cv2.contourArea(contour)  # Calculate area in pixels
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw contour
        label_text = f"{w}px x {h}px"
        cv2.putText(
            image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
        )  # Increased font size

# Show the result
cv2.imshow("Filtered Cells with Larger Labels", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
