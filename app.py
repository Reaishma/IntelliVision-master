import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import image_classification
import object_detection
import image_segmentation
import feature_detection
import style_transfer
import image_enhancement
import edge_detection
from image_utils import load_image, save_image, display_images_side_by_side

def main():
    st.set_page_config(
        page_title="Computer Vision Application",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔍 Computer Vision Application")
    st.markdown("### Advanced Image Processing and Analysis Platform")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Computer Vision Tasks")
    task = st.sidebar.selectbox(
        "Choose a CV Task:",
        [
            "Image Classification",
            "Object Detection",
            "Image Segmentation", 
            "Feature Detection",
            "Style Transfer",
            "Image Enhancement",
            "Edge Detection"
        ]
    )
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image to process"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = load_image(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.info(f"""
            **Image Information:**
            - Size: {image.size}
            - Mode: {image.mode}
            - Format: {uploaded_file.type}
            """)
        
        with col2:
            st.subheader(f"Results: {task}")
            
            # Route to appropriate CV module
            if task == "Image Classification":
                image_classification.run(image)
            elif task == "Object Detection":
                object_detection.run(image)
            elif task == "Image Segmentation":
                image_segmentation.run(image)
            elif task == "Feature Detection":
                feature_detection.run(image)
            elif task == "Style Transfer":
                style_transfer.run(image)
            elif task == "Image Enhancement":
                image_enhancement.run(image)
            elif task == "Edge Detection":
                edge_detection.run(image)
    
    else:
        st.info("👆 Please upload an image to begin processing")
        
        # Show sample capabilities
        st.markdown("## 🚀 Available Computer Vision Tasks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Image Analysis**
            - Image Classification
            - Object Detection
            - Feature Detection
            """)
            
        with col2:
            st.markdown("""
            **🎨 Image Processing**
            - Style Transfer
            - Image Enhancement
            - Edge Detection
            """)
            
        with col3:
            st.markdown("""
            **🧠 Advanced CV**
            - Image Segmentation
            - Contour Analysis
            - Pattern Recognition
            """)

if __name__ == "__main__":
    main()
