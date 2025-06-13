import cv2
import numpy as np

# Load the image
image = cv2.imread("cell_image.jpg")

# Convert to grayscale for better accuracy
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding for better segmentation
thresholded = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Find cell contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and add width labels
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw contour
    cv2.putText(
        image, f"{w}px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )  # Add width label

# Show the processed image
cv2.imshow("Cells with Width Labels", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
