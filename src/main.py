import cv2
import matplotlib.pyplot as plt

# Input image
img = "../asset/rocky.jpeg"

# Process image to detect edges using various tools from OpenCV
image_ori = cv2.imread(img, cv2.IMREAD_COLOR)
image_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient_magnitude = cv2.magnitude(sobelx, sobely)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

cv2.imwrite("../asset/rocky_sobel.jpeg", gradient_magnitude)

cv2.imshow("image", gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_img = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
sobel_img = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(rgb_img)

plt.subplot(1,2,2)
plt.title("Sobel")
plt.imshow(sobel_img)

plt.show()