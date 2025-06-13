import cv2
import numpy as np
import random
from skimage import io, color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

# Load the image
image = io.imread("particles_image.png", as_gray=True)

# Apply edge detection (Canny filter)
edges = canny(image, sigma=2)

# Define range for particle detection (between 50px and 100px diameter)
hough_radii = np.arange(25, 50, 2)
hough_res = hough_circle(edges, hough_radii)

# Find the most prominent circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii)

# Convert to color image for visualization
image_color = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Filter only particles with diameter in the valid range
valid_circles = [(x, y, r) for x, y, r in zip(cx, cy, radii) if 50 <= 2 * r <= 80]

# Randomly select 10 particles
random.shuffle(valid_circles)
selected_circles = valid_circles[:10]

# Draw and label selected particles
for x, y, r in selected_circles:
    diameter = 2 * r
    cv2.circle(image_color, (int(x), int(y)), int(r), (0, 255, 0), 2)  # Draw circle
    label_text = f"D={diameter}px"
    cv2.putText(
        image_color,
        label_text,
        (int(x) - int(r), int(y) - int(r) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )  # Minimized label size

# Show result
cv2.imshow("Random 10 Particles with Labels", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
