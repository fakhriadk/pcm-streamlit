# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import exposure, img_as_ubyte
from PIL import Image

# === Helper functions ===

def plot_histogram(image):
    hist = ndi.histogram(image, min=0, max=255, bins=256)
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_title("Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def histogram_equalization(image):
    hist = ndi.histogram(image, min=0, max=255, bins=256)
    cdf = hist.cumsum() / hist.sum()
    im_eq = cdf[image] * 1
    im_eq = img_as_ubyte(im_eq)
    return im_eq

def adaptive_equalization(image, clip_limit=0.03):
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    return img_as_ubyte(img_adapteq)

def mean_filter(image, kernel_size=3):
    kernel = np.full((kernel_size, kernel_size), 1 / (kernel_size ** 2))
    if image.ndim == 3:
        image = image[:, :, 0]  # ambil channel pertama aja
    filtered = ndi.convolve(image, kernel)
    return filtered

# === Streamlit App ===

st.set_page_config(page_title="Image Processing App", layout="wide")
st.title("Muhammad Fakhri Andika Mutiara (5023211056) - Program Assignment 1")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Histogram")
        plot_histogram(image_np)

    st.markdown("---")

    st.subheader("Histogram Equalization")
    im_eq = histogram_equalization(image_np)
    st.image(im_eq, caption="Histogram Equalized", use_container_width=True)

    st.subheader("Adaptive Histogram Equalization")
    clip_limit = st.slider("Clip Limit", min_value=0.01, max_value=0.1, value=0.03, step=0.01)
    im_adapt = adaptive_equalization(image_np, clip_limit=clip_limit)
    st.image(im_adapt, caption="Adaptive Equalized", use_container_width=True)

    st.subheader("Mean Filter")
    kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
    im_filt = mean_filter(image_np, kernel_size)
    st.image(im_filt, caption=f"Mean Filtered (kernel size {kernel_size})", use_container_width=True)
