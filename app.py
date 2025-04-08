# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import exposure, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
from PIL import Image

# === Helper Functions ===

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
    equalized = cdf[image]
    return img_as_ubyte(equalized)

def adaptive_equalization(image, clip_limit=0.03):
    adap_eq = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    return img_as_ubyte(adap_eq)

def mean_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    if image.ndim == 3:
        image = image[:, :, 0]
    return ndi.convolve(image, kernel)

def gaussian_filter(image, sigma=1.0):
    return ndi.gaussian_filter(image, sigma=sigma)

def median_filter(image, size=3):
    return ndi.median_filter(image, size=size)

def calculate_metrics(original, restored):
    return psnr(original, restored), mse(original, restored)

# === Streamlit App ===

st.set_page_config(page_title="Image Processing App", layout="wide")
st.title("Muhammad Fakhri Andika Mutiara (5023211056) - Assignment 1")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original Image")
        st.image(image, width=300)
    with col2:
        st.subheader("Histogram")
        plot_histogram(image_np)

    st.markdown("---")

    with st.expander("Histogram Equalization"):
        eq_img = histogram_equalization(image_np)
        st.image(eq_img, width=300, caption="Equalized Image")

    with st.expander("Adaptive Histogram Equalization (CLAHE)"):
        clip_limit = st.slider("Clip Limit", 0.01, 0.1, 0.03, step=0.01)
        adapt_img = adaptive_equalization(image_np / 255.0, clip_limit)
        st.image(adapt_img, width=300, caption="Adaptive Equalized")

    with st.expander("Mean Filter"):
        kernel_size = st.slider("Kernel Size (Mean)", 3, 15, step=2, value=3)
        mean_img = mean_filter(image_np, kernel_size)
        st.image(mean_img, width=300, caption=f"Mean Filtered (Kernel Size {kernel_size})")

    with st.expander("Gaussian Filter (Linear Restoration)"):
        sigma = st.slider("Sigma (Gaussian)", 0.5, 5.0, step=0.5, value=1.0)
        gauss_img = gaussian_filter(image_np, sigma)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(gauss_img, width=300, caption=f"Gaussian (σ = {sigma})")
        with col2:
            psnr_val, mse_val = calculate_metrics(image_np, gauss_img)
            st.metric("PSNR", f"{psnr_val:.2f} dB")
            st.metric("MSE", f"{mse_val:.2f}")

    with st.expander("Median Filter (Non-Linear Restoration)"):
        median_size = st.slider("Kernel Size (Median)", 3, 15, step=2, value=3)
        median_img = median_filter(image_np, median_size)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(median_img, width=300, caption=f"Median (Kernel Size {median_size})")
        with col2:
            psnr_med, mse_med = calculate_metrics(image_np, median_img)
            st.metric("PSNR", f"{psnr_med:.2f} dB")
            st.metric("MSE", f"{mse_med:.2f}")
