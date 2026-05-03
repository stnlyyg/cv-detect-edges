import cv2
import numpy as np
import streamlit as st


# Edge detection helpers
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

def process_image(method, uploaded_file):
    if uploaded_file is None:
        return None, None

    # read uploaded file bytes and decode with OpenCV
    file_bytes = uploaded_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_ori = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_ori is None:
        return None, None

    image_gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)

    if method.lower() == "sobel":
        result = sobel_method(image_gray)
    elif method.lower() == "laplacian":
        result = laplacian_method(image_gray)
    else:
        result = image_gray

    return image_ori, result


def main():
    st.title("Edge Detection using OpenCV")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    method = st.radio(
        "Choose an edge detection method",
        options=["Sobel", "Laplacian"],
    )

    if uploaded_file is None:
        st.info("Please upload an image to see results.")
        return

    image_ori, result = process_image(method, uploaded_file)
    if image_ori is None or result is None:
        st.error("Could not read the uploaded image.")
        return

    # convert images for display (OpenCV uses BGR)
    original_rgb = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    col1, col2 = st.columns(2)
    col1.image(original_rgb, caption="Original", width="stretch")
    col2.image(processed_rgb, caption=method, width="stretch")


if __name__ == "__main__":
    main()