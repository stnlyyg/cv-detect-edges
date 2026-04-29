import cv2
import matplotlib.pyplot as plt

# Input image
# img = "../asset/rocky.jpeg"

# Process image to detect edges using various tools from OpenCV
def convert_to_gray(ori_image):
    image_ori = cv2.imread(ori_image, cv2.IMREAD_COLOR)
    image_gray = cv2.imread(ori_image, cv2.IMREAD_GRAYSCALE)
    return image_ori, image_gray

def sobel_method(image_gray):
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)

    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    result_sobel = cv2.convertScaleAbs(gradient_magnitude)

    return result_sobel

def laplacian_method(image_gray):
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    return laplacian_abs

def main(method, img_path):
    image_ori, image_gray = convert_to_gray(img_path)

    if method.lower() == "sobel":
        result = sobel_method(image_gray)
    elif method.lower() == "laplacian":
        result = laplacian_method(image_gray)

    # cv2.imwrite(f"../asset/rocky_{method}.jpeg", result)

    rgb_img = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    processed_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(rgb_img, cmap='gray')

    plt.subplot(1,2,2)
    plt.title(method.capitalize())
    plt.imshow(processed_img, cmap='gray')

    plt.savefig(f"../asset/rocky_{method}.jpeg")
    plt.show()

if __name__ == "__main__":
    chosen_method = input("Choose an edge detection method [sobel/laplacian]: ")
    image_path = input("Paste image path: ")
    main(chosen_method, image_path)