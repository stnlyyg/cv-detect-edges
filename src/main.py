import cv2
import matplotlib.pyplot as plt

# Input image
img = "../asset/rocky.jpeg"

image = cv2.imread(img)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Process image to detect edges using various tools from OpenCV