# Import necessary libraries
import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import scipy.ndimage as ndi
from skimage import morphology
from skimage.measure import label, regionprops_table
from skimage.measure import regionprops
import pandas as pd
import tempfile
import os
import math  

# Function to display the slices of the image
def display_slices(data):
    fig, axes = plt.subplots(6, 3, figsize=(10, 15))
    slice_counter = 0
    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(data[:, :, slice_counter].T, cmap="gray", origin="lower")
            axes[i, j].set_title(f"Slice {slice_counter}")
            axes[i, j].axis("off")
            slice_counter += 1
    for i in range(3, 6):
        for j in range(3):
            axes[i, j].imshow(data[:, :, slice_counter].T, cmap="gray", origin="lower")
            axes[i, j].set_title(f"Slice {slice_counter}")
            axes[i, j].axis("off")
            slice_counter += 1
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit UI
st.title("5023211056_Muhammad Fakhri Andika Mutiara_Assignment Masking")

# File uploader for NIfTI file
nifti_file = st.file_uploader("Upload NIfTI file (.nii)", type=["nii"])

if nifti_file is not None:
    # Save the uploaded file to a temporary location with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(nifti_file.read())
        temp_file_path = temp_file.name
    
    # Load the NIfTI image using nibabel
    try:
        nifti = nib.load(temp_file_path)
        data = nifti.get_fdata()

            # Obtain pixdim and round it
        pixdim = nifti.header['pixdim']
        rounded_pixdim = np.round(pixdim, 4)  # Rounding to 4 decimal places for readability


        # Show header and image shape
        st.subheader("NIfTI Image Information")
        st.write(f"Shape: {data.shape}")

        
        # Display header info in a more readable format
        nifti_header = nifti.header
        header_info = {
            
            "sizeof_hdr": nifti_header['sizeof_hdr'],
            "data_type": nifti_header['data_type'],
            "db_name": nifti_header['db_name'],
            "extents": nifti_header['extents'],
            "session_error": nifti_header['session_error'],
            "regular": nifti_header['regular'],
            "dim_info": nifti_header['dim_info'],
            "dim": nifti_header['dim'],
            "intent_p1": nifti_header['intent_p1'],
            "intent_p2": nifti_header['intent_p2'],
            "intent_p3": nifti_header['intent_p3'],
            "intent_code": nifti_header['intent_code'],
            "datatype": nifti_header['datatype'],
            "bitpix": nifti_header['bitpix'],
            "slice_start": nifti_header['slice_start'],
            "pixdim": rounded_pixdim,
            "vox_offset": nifti_header['vox_offset'],
            "scl_slope": nifti_header['scl_slope'],
            "scl_inter": nifti_header['scl_inter'],
            "slice_end": nifti_header['slice_end'],
            "slice_code": nifti_header['slice_code'],
            "xyzt_units": nifti_header['xyzt_units'],
            "cal_max": nifti_header['cal_max'],
            "cal_min": nifti_header['cal_min'],
            "slice_duration": nifti_header['slice_duration'],
            "toffset": nifti_header['toffset'],
            "glmax": nifti_header['glmax'],
            "glmin": nifti_header['glmin'],
            "descrip": nifti_header['descrip'],
            "aux_file": nifti_header['aux_file'],
            "qform_code": nifti_header['qform_code'],
            "sform_code": nifti_header['sform_code'],
            "quatern_b": nifti_header['quatern_b'],
            "quatern_c": nifti_header['quatern_c'],
            "quatern_d": nifti_header['quatern_d'],
            "qoffset_x": nifti_header['qoffset_x'],
            "qoffset_y": nifti_header['qoffset_y'],
            "qoffset_z": nifti_header['qoffset_z'],
            "srow_x": nifti_header['srow_x'],
            "srow_y": nifti_header['srow_y'],
            "srow_z": nifti_header['srow_z'],
            "intent_name": nifti_header['intent_name'],
            "magic": nifti_header['magic']
        }
        
        for key, value in header_info.items():
            st.write(f"{key: <15}: {value}")

        # Obtain pixel data
        data = nifti.get_fdata()

        # Display pixel data dtype and shape
        st.subheader("Pixel Data Information")
        st.write(f"Pixel Data Type: {data.dtype}")
        st.write(f"Pixel Data Shape: {data.shape}")
        # Button to display slices
        if st.button("Show Slices"):
            display_slices(data)

        # Select a slice to process (default: slice 14)
        slice_index = st.slider("Select slice to display", min_value=0, max_value=data.shape[2]-1, value=14)
        imtype = data[:, :, slice_index]


        # Display the selected slice with colorbar
        st.subheader(f"Slice {slice_index}")
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(imtype, cmap='gray', vmin=0, vmax=255)
        ax.axis("off")
        st.pyplot(fig)

        
        # Display data type and min/max values
        st.write(f"Data type: {imtype.dtype}")
        st.write(f"Min. value: {imtype.min()}")
        st.write(f"Max. value: {imtype.max()}")
        st.write(f"Shape: {imtype.shape}")





        # Normalize the image slice to 0-255 and convert to uint8
        imtype_norm = ((imtype - np.min(imtype)) / (np.max(imtype) - np.min(imtype)) * 255).astype(np.uint8)

        # Display the normalized slice
        st.subheader("Normalized Image Slice")
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(imtype_norm.T, cmap='gray', origin='lower', vmin=0, vmax=255)
        fig.colorbar(cax)  # Add colorbar
        st.pyplot(fig)

        # Display data type and min/max values of the normalized image
        st.write(f"Data type: {imtype_norm.dtype}")
        st.write(f"Min. value: {imtype_norm.min()}")
        st.write(f"Max. value: {imtype_norm.max()}")
        st.write(f"Shape: {imtype_norm.shape}")

        # Print normalized image info in the console
        print("Data type:", imtype_norm.dtype)
        print("Shape:", imtype_norm.shape)
        print("Min value:", imtype_norm.min())
        print("Max value:", imtype_norm.max())

        

        # Button to show histogram and CDF
        if st.button("Show Histogram and CDF"):
        # Compute histogram
            hist_norm, bin_edges = np.histogram(imtype_norm.flatten(), bins=256, range=(0, 255))
    
        #   Compute CDF
            cdf_normalized = hist_norm.cumsum()  # Cumulative sum of the histogram
            cdf_normalized = cdf_normalized / cdf_normalized[-1]  # Normalize the CDF

        # Plot histogram and CDF for normalized image
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

        # Plot histogram
            axes[0].plot(bin_edges[:-1], hist_norm, color='blue')
            axes[0].set_title("Intensity Histogram")
        #  axes[0].set_xlabel("Pixel Intensity")
            axes[0].set_ylabel("Frequency")

        # Plot CDF
            axes[1].plot(bin_edges[:-1], cdf_normalized, color='red')
            axes[1].set_title("Cumulative Distribution Function (CDF)")
            axes[1].set_xlabel("Pixel Intensity")
            axes[1].set_ylabel("Cumulative Probability")

            st.pyplot(fig)  # Show the plots






                # Apply CLAHE to the image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        slice_clahe = clahe.apply(imtype_norm)

        # Display CLAHE-enhanced image
        st.subheader("CLAHE Enhanced Image")
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(slice_clahe.T, cmap='gray', origin='lower')
        fig.colorbar(cax)  # Add colorbar to the CLAHE image
        ax.axis("off")
        st.pyplot(fig)

        # Button to show histogram and CDF of CLAHE-enhanced image
        if st.button("Show Histogram and CDF of CLAHE Image"):
            # Compute histogram for the CLAHE-enhanced image
            hist_clahe, bin_edges_clahe = np.histogram(slice_clahe.flatten(), bins=256, range=(0, 255))
            
            # Compute CDF for the CLAHE-enhanced image
            cdf_clahe_normalized = hist_clahe.cumsum()  # Cumulative sum of the histogram
            cdf_clahe_normalized = cdf_clahe_normalized / cdf_clahe_normalized[-1]  # Normalize the CDF

            # Plot histogram and CDF for CLAHE-enhanced image
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

            # Plot histogram for CLAHE-enhanced image
            axes[0].plot(bin_edges_clahe[:-1], hist_clahe, color='blue')
            axes[0].set_title("Intensity Histogram of CLAHE Image")
            axes[0].set_xlabel("Pixel Intensity")
            axes[0].set_ylabel("Frequency")

            # Plot CDF for CLAHE-enhanced image
            axes[1].plot(bin_edges_clahe[:-1], cdf_clahe_normalized, color='red')
            axes[1].set_title("Cumulative Distribution Function (CDF) of CLAHE Image")
            axes[1].set_xlabel("Pixel Intensity")
            axes[1].set_ylabel("Cumulative Probability")

            st.pyplot(fig)  # Show the plots






        # Ambil gambar hasil CLAHE
        image = slice_clahe

        # Slider to adjust threshold value
        manual_threshold = st.slider("Select threshold for manual binarization", min_value=0, max_value=255, value=110)

        # Step 1: Binarisasi dengan threshold manual
        binary_image_manual = image >= manual_threshold

        # Plot gambar + garis contour di threshold level
        st.subheader(f"Manual Thresholding at {manual_threshold}")
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(image.T, cmap='gray', origin='lower')
        ax.contour(image.T, levels=[manual_threshold], colors='red')  # tampilkan contour pada level threshold
        ax.set_title(f"Manual Threshold = {manual_threshold}")
        ax.axis('off')
        fig.colorbar(cax)  # Add colorbar
        st.pyplot(fig)

        # Button to apply binary thresholding and show morphological operations
        if st.button("Apply Threshold and Morphological Operations"):
            # Binary thresholding
            binary_image_manual = image >= manual_threshold

            # Remove small objects and fill holes
            only_large_blobs = morphology.remove_small_objects(binary_image_manual, min_size=40)
            filled_blobs = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=300))

            # Display the result after morphological operations
            st.subheader("Morphological Operations Result")
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(filled_blobs.T, cmap='gray', origin='lower')
            ax.set_title("After Morphological Operations")
            ax.axis("off")
            st.pyplot(fig)
            
                



        # Binary thresholding and morphological operations
        manual_threshold = st.slider("Select threshold", min_value=0, max_value=255, value=110)
        binary_image_manual = slice_clahe >= manual_threshold

        # Remove small objects and fill holes
        only_large_blobs = morphology.remove_small_objects(binary_image_manual, min_size=40)
        filled_blobs = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large_blobs), min_size=300))

        # Labeling and region properties
        labels, nlabels = ndi.label(filled_blobs)

        # Display labeled components
        st.subheader(f"Labeled Components ({nlabels} Areas)")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(labels.T, cmap='tab20', origin='lower')
        ax.axis("off")
        st.pyplot(fig)

        # Step 6: Remove small objects from the segmentation
        image_segmented = filled_blobs.copy()  # Starting with the cleaned image
        for label_ind, label_coords in enumerate(np.unique(labels)):
            if label_coords == 0:
                continue  # Skip background
            cell = image_segmented[labels == label_coords]
            if np.prod(cell.shape) < 2000:  # Removing small objects
                st.write(f"Label {label_ind} is too small! Setting to 0.")
                image_segmented = np.where(labels == label_coords, 0, image_segmented)

        # Step 7: Relabel the image after small object removal
        labels, nlabels = ndi.label(image_segmented)
        st.write(f"There are now {nlabels} objects detected.")



        
        # Assuming `image_segmented` is your segmented image
        image = image_segmented

        # Step 1: Rotate the image 90 degrees counterclockwise
        image_rotated = np.rot90(image, k=1)  # k=1 for counterclockwise 90-degree rotation

        # Step 2: Label the rotated image
        label_img = label(image_rotated)

        # Step 3: Extract region properties
        regions = regionprops(label_img)

        # Step 4: Create a colormap for distinct colors for each region
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Step 5: Visualize with unique colors
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(image_rotated, cmap=plt.cm.gray)

        for idx, props in enumerate(regions):
            # Get the centroid and orientation of the region
            y0, x0 = props.centroid
            orientation = props.orientation

            # Calculate the endpoints of the orientation lines
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

            # Choose a color for this region
            color = colors[idx % len(colors)]

            # Plot the orientation lines within the bounding box (constrained to box)
            minr, minc, maxr, maxc = props.bbox
            x1 = np.clip(x1, minc, maxc)
            y1 = np.clip(y1, minr, maxr)
            x2 = np.clip(x2, minc, maxc)
            y2 = np.clip(y2, minr, maxr)

            # Plot orientation lines in the selected color
            ax.plot((x0, x1), (y0, y1), color=color, linewidth=2.5)
            ax.plot((x0, x2), (y0, y2), color=color, linewidth=2.5)

            # Plot the centroid (center line)
            ax.plot(x0, y0, '.', color='red', markersize=15)

            # Plot the bounding box in a darker color for contrast
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, color=color, linewidth=2.5)

        ax.axis('off')  # Hide axes for better visualization
        st.pyplot(fig)


        

        # Extract unique labels (excluding background)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0][:6]  # Exclude background, take first 6 labels

        # Show isolated blobs for the first 6 unique labels
        fig, axes = plt.subplots(nrows=1, ncols=len(unique_labels), figsize=(12, 6))

        for ii, label_id in enumerate(unique_labels):
            obj_mask = (labels == label_id).astype(np.uint8)
            isolated_blob = image_segmented * obj_mask

            axes[ii].imshow(isolated_blob.T, cmap='gray', origin='lower')
            axes[ii].axis('off')
            axes[ii].set_title(f"Label #{label_id}\nSize: {obj_mask.sum()}")

        plt.tight_layout()
        st.pyplot(fig)

        # Export region properties to DataFrame
        region_props = regionprops_table(labels, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length'))
        props_df = pd.DataFrame(region_props)





        # Display DataFrame
        st.subheader("Region Properties")
        st.write(props_df)

        # Button to download the DataFrame as Excel
        if st.button("Download Region Properties as Excel"):
            excel_filename = "region_properties.xlsx"
            props_df.to_excel(excel_filename, index=False)

            # Create a download link
            with open(excel_filename, "rb") as f:
                st.download_button("Download Excel File", data=f, file_name=excel_filename)

    except Exception as e:
        st.error(f"Error loading NIfTI file: {e}")
    
    # Clean up temporary file
    os.remove(temp_file_path)








