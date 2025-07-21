import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

def detect_harris_corners(image, threshold=0.01, k=0.04):
    """Harris Corner Detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Harris corner detection
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, k)
    
    # Threshold for optimal value
    corners = np.where(dst > threshold * dst.max())
    
    return corners, dst

def detect_sift_features(image, nfeatures=500):
    """SIFT Feature Detection"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Create SIFT detector
        sift = cv2.SIFT_create(nFeatures=nfeatures)
        
        # Detect keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    except Exception as e:
        st.warning(f"SIFT not available: {str(e)}. Using ORB instead.")
        return detect_orb_features(image, nfeatures)

def detect_orb_features(image, nfeatures=500):
    """ORB Feature Detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(nFeatures=nfeatures)
    
    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def detect_fast_corners(image, threshold=10):
    """FAST Corner Detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Create FAST detector
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    
    # Detect keypoints
    keypoints = fast.detect(gray, None)
    
    return keypoints

def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """Canny Edge Detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges

def detect_contours(image, threshold=127):
    """Contour Detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, thresh

def draw_harris_corners(image, corners):
    """Draw Harris corners on image"""
    img_with_corners = image.copy()
    img_array = np.array(img_with_corners)
    
    # Mark detected corners in red
    img_array[corners] = [255, 0, 0]
    
    return Image.fromarray(img_array)

def draw_keypoints(image, keypoints, feature_type="SIFT"):
    """Draw keypoints on image"""
    img_array = np.array(image)
    
    # Convert keypoints to image coordinates
    img_with_keypoints = img_array.copy()
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_with_keypoints, (x, y), 3, (255, 0, 0), -1)
        
        # Draw orientation for SIFT
        if feature_type == "SIFT" and hasattr(kp, 'angle'):
            length = int(kp.size)
            angle = np.radians(kp.angle)
            x2 = int(x + length * np.cos(angle))
            y2 = int(y + length * np.sin(angle))
            cv2.line(img_with_keypoints, (x, y), (x2, y2), (0, 255, 0), 1)
    
    return Image.fromarray(img_with_keypoints)

def draw_contours_on_image(image, contours):
    """Draw contours on image"""
    img_array = np.array(image)
    
    # Draw contours
    cv2.drawContours(img_array, contours, -1, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

def run(image):
    """Run feature detection"""
    st.markdown("### 🔍 Feature Detection")
    
    # Feature detection method selection
    method = st.selectbox(
        "Detection Method:",
        ["Harris Corner Detection", "SIFT Features", "ORB Features", "FAST Corners", "Canny Edges", "Contours"]
    )
    
    # Method-specific parameters
    if method == "Harris Corner Detection":
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Corner Threshold:", 0.001, 0.1, 0.01, 0.001)
        with col2:
            k = st.slider("Harris Parameter k:", 0.01, 0.1, 0.04, 0.01)
        params = {'threshold': threshold, 'k': k}
        
    elif method in ["SIFT Features", "ORB Features"]:
        nfeatures = st.slider("Maximum Features:", 100, 2000, 500, 50)
        params = {'nfeatures': nfeatures}
        
    elif method == "FAST Corners":
        threshold = st.slider("FAST Threshold:", 1, 50, 10)
        params = {'threshold': threshold}
        
    elif method == "Canny Edges":
        col1, col2 = st.columns(2)
        with col1:
            low_threshold = st.slider("Low Threshold:", 1, 100, 50)
        with col2:
            high_threshold = st.slider("High Threshold:", 100, 300, 150)
        params = {'low_threshold': low_threshold, 'high_threshold': high_threshold}
        
    elif method == "Contours":
        threshold = st.slider("Binary Threshold:", 0, 255, 127)
        params = {'threshold': threshold}
    
    if st.button("🚀 Detect Features", type="primary"):
        with st.spinner(f"Detecting features using {method}..."):
            
            try:
                if method == "Harris Corner Detection":
                    corners, dst = detect_harris_corners(image, **params)
                    result_image = draw_harris_corners(image, corners)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image, caption="Input Image", use_column_width=True)
                    
                    with col2:
                        st.subheader("Corner Response")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(dst, cmap='hot')
                        ax.set_title('Harris Corner Response')
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    with col3:
                        st.subheader("Detected Corners")
                        st.image(result_image, caption="Harris Corners", use_column_width=True)
                    
                    # Statistics
                    num_corners = len(corners[0])
                    st.metric("Corners Detected", num_corners)
                
                elif method in ["SIFT Features", "ORB Features"]:
                    if method == "SIFT Features":
                        keypoints, descriptors = detect_sift_features(image, **params)
                        feature_type = "SIFT"
                    else:
                        keypoints, descriptors = detect_orb_features(image, **params)
                        feature_type = "ORB"
                    
                    result_image = draw_keypoints(image, keypoints, feature_type)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image, caption="Input Image", use_column_width=True)
                    
                    with col2:
                        st.subheader(f"{feature_type} Features")
                        st.image(result_image, caption=f"{feature_type} Keypoints", use_column_width=True)
                    
                    # Feature statistics
                    st.subheader("📊 Feature Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Features Detected", len(keypoints))
                    
                    with col2:
                        if descriptors is not None:
                            st.metric("Descriptor Size", descriptors.shape[1] if len(keypoints) > 0 else 0)
                        else:
                            st.metric("Descriptor Size", "N/A")
                    
                    with col3:
                        avg_response = np.mean([kp.response for kp in keypoints]) if keypoints else 0
                        st.metric("Avg Response", f"{avg_response:.3f}")
                    
                    # Feature distribution
                    if keypoints:
                        st.subheader("📈 Feature Distribution")
                        
                        # Response distribution
                        responses = [kp.response for kp in keypoints]
                        sizes = [kp.size for kp in keypoints]
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        ax1.hist(responses, bins=20, alpha=0.7, color='#667eea')
                        ax1.set_xlabel('Feature Response')
                        ax1.set_ylabel('Count')
                        ax1.set_title('Feature Response Distribution')
                        ax1.grid(True, alpha=0.3)
                        
                        ax2.hist(sizes, bins=20, alpha=0.7, color='#764ba2')
                        ax2.set_xlabel('Feature Size')
                        ax2.set_ylabel('Count')
                        ax2.set_title('Feature Size Distribution')
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                elif method == "FAST Corners":
                    keypoints = detect_fast_corners(image, **params)
                    result_image = draw_keypoints(image, keypoints, "FAST")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image, caption="Input Image", use_column_width=True)
                    
                    with col2:
                        st.subheader("FAST Corners")
                        st.image(result_image, caption="FAST Keypoints", use_column_width=True)
                    
                    st.metric("FAST Corners Detected", len(keypoints))
                
                elif method == "Canny Edges":
                    edges = detect_edges_canny(image, **params)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image, caption="Input Image", use_column_width=True)
                    
                    with col2:
                        st.subheader("Canny Edges")
                        st.image(edges, caption="Edge Detection", use_column_width=True, clamp=True)
                    
                    # Edge statistics
                    edge_pixels = np.sum(edges > 0)
                    total_pixels = edges.size
                    edge_percentage = (edge_pixels / total_pixels) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Edge Pixels", edge_pixels)
                    with col2:
                        st.metric("Edge Percentage", f"{edge_percentage:.2f}%")
                
                elif method == "Contours":
                    contours, thresh = detect_contours(image, **params)
                    result_image = draw_contours_on_image(image, contours)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image, caption="Input Image", use_column_width=True)
                    
                    with col2:
                        st.subheader("Thresholded")
                        st.image(thresh, caption="Binary Image", use_column_width=True, clamp=True)
                    
                    with col3:
                        st.subheader("Contours")
                        st.image(result_image, caption="Detected Contours", use_column_width=True)
                    
                    # Contour statistics
                    st.subheader("📊 Contour Analysis")
                    
                    if contours:
                        areas = [cv2.contourArea(contour) for contour in contours]
                        perimeters = [cv2.arcLength(contour, True) for contour in contours]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Contours", len(contours))
                        
                        with col2:
                            st.metric("Max Area", f"{max(areas):.0f}" if areas else "0")
                        
                        with col3:
                            st.metric("Avg Area", f"{np.mean(areas):.0f}" if areas else "0")
                        
                        with col4:
                            st.metric("Avg Perimeter", f"{np.mean(perimeters):.0f}" if perimeters else "0")
                        
                        # Area distribution
                        if len(areas) > 1:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.hist(areas, bins=min(20, len(areas)), alpha=0.7, color='#667eea')
                            ax.set_xlabel('Contour Area (pixels)')
                            ax.set_ylabel('Count')
                            ax.set_title('Contour Area Distribution')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning("No contours detected. Try adjusting the threshold.")
                
            except Exception as e:
                st.error(f"Error during feature detection: {str(e)}")
                st.info("Please try with different parameters or a different image.")
