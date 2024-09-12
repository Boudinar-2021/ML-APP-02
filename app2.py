import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

st.set_page_config(page_title='Medical Image Filter App', layout='wide')

st.markdown("""
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def zip_images(images_dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for filename, image in images_dict.items():
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            zip_file.writestr(f"{filename}.png", img_byte_arr.getvalue())
    buffer.seek(0)
    return buffer

st.sidebar.header('1. Upload your Input Image')
uploaded_file = st.sidebar.file_uploader("Upload your input image", type=["png", "jpg", "jpeg"])

st.sidebar.header('2. Select Filters to Apply')

# Adding help to checkboxes
blur = st.sidebar.checkbox('Blur', help='Apply a simple blur effect using a box filter.')
median_blur = st.sidebar.checkbox('Median Blur', help='Apply a median blur, which is good for removing salt-and-pepper noise.')
gaussian_filter = st.sidebar.checkbox('Gaussian Filter', help='Apply a Gaussian filter for smoothing images.')
canny_algorithm = st.sidebar.checkbox('Canny Algorithm', help='Apply the Canny edge detection algorithm to the image.')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown('<h2 class="centered-text">Original Image</h2>', unsafe_allow_html=True)
    st.image(image, caption='Original Image', use_column_width=False)  # Display original size

    img_cv2 = pil_to_cv2(image)

    images_dict = {'Original Image': image}

    if blur:
        blur_ksize = st.sidebar.slider('Blur Kernel Size', min_value=3, max_value=30, value=5, step=2, 
                                       help='Kernel size for the blur effect. Recommended value: 5.')
        st.markdown(f'<h2 class="centered-text">Blurred Image</h2>', unsafe_allow_html=True)
        img_blur = cv2.blur(img_cv2, (blur_ksize, blur_ksize))
        images_dict['Blurred Image'] = cv2_to_pil(img_blur)
        st.image(cv2_to_pil(img_blur), caption=f'Blurred Image (Kernel Size: {blur_ksize})', use_column_width=False)

    if median_blur:
        median_ksize = st.sidebar.slider('Median Blur Kernel Size', min_value=3, max_value=30, value=5, step=2, 
                                         help='Kernel size for median blur. Recommended value: 5.')
        st.markdown(f'<h2 class="centered-text">Median Blurred Image</h2>', unsafe_allow_html=True)
        img_median = cv2.medianBlur(img_cv2, median_ksize)
        images_dict['Median Blurred Image'] = cv2_to_pil(img_median)
        st.image(cv2_to_pil(img_median), caption=f'Median Blurred Image (Kernel Size: {median_ksize})', use_column_width=False)

    if gaussian_filter:
        gaussian_ksize = st.sidebar.slider('Gaussian Kernel Size', min_value=3, max_value=30, value=5, step=2, 
                                           help='Kernel size for Gaussian blur. Recommended value: 5.')
        gaussian_sigma = st.sidebar.slider('Gaussian Sigma', min_value=0.1, max_value=10.0, value=1.0, step=0.1, 
                                           help='Sigma for Gaussian blur. Recommended value: 1.0.')
        st.markdown(f'<h2 class="centered-text">Gaussian Filtered Image</h2>', unsafe_allow_html=True)
        img_gaussian = cv2.GaussianBlur(img_cv2, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
        images_dict['Gaussian Filtered Image'] = cv2_to_pil(img_gaussian)
        st.image(cv2_to_pil(img_gaussian), caption=f'Gaussian Filtered Image (Kernel Size: {gaussian_ksize}, Sigma: {gaussian_sigma})', use_column_width=False)

    if canny_algorithm:
        canny_thresh1 = st.sidebar.slider('Canny Threshold 1', min_value=0, max_value=255, value=100, 
                                          help='First threshold for the hysteresis in the Canny edge detection. Recommended value: 100.')
        canny_thresh2 = st.sidebar.slider('Canny Threshold 2', min_value=0, max_value=255, value=200, 
                                          help='Second threshold for the hysteresis in the Canny edge detection. Recommended value: 200.')
        st.markdown(f'<h2 class="centered-text">Canny Algorithm Applied</h2>', unsafe_allow_html=True)
        img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_gray, canny_thresh1, canny_thresh2)
        images_dict['Canny Algorithm Applied'] = cv2_to_pil(img_canny)
        st.image(cv2_to_pil(img_canny), caption=f'Canny Algorithm Applied (Thresholds: {canny_thresh1}, {canny_thresh2})', use_column_width=False)

    zip_buffer = zip_images(images_dict)

    st.download_button(
        label="⬇️ Download All Processed Images",
        data=zip_buffer,
        file_name="processed_images.zip",
        mime="application/zip"
    )
else:
    st.info('Awaiting for Medical Image to be uploaded.')
