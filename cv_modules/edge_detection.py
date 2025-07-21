import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage import feature, filters, morphology
from scipy import ndimage

def canny_edge_detection(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """Canny edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    
    return edges

def sobel_edge_detection(image, direction='both'):
    """Sobel edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    if direction == 'x':
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == 'y':
        edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    elif direction == 'both':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges

def laplacian_edge_detection(image, kernel_size=3):
    """Laplacian edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    
    # Convert to absolute values and normalize
    edges = np.absolute(laplacian)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges

def prewitt_edge_detection(image):
    """Prewitt edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Apply kernels
    edges_x = cv2.filter2D(gray, -1, kernel_x)
    edges_y = cv2.filter2D(gray, -1, kernel_y)
    
    # Combine
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges

def roberts_edge_detection(image):
    """Roberts cross-gradient edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Roberts kernels
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    # Apply kernels
    edges_x = cv2.filter2D(gray, -1, kernel_x)
    edges_y = cv2.filter2D(gray, -1, kernel_y)
    
    # Combine
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges

def scharr_edge_detection(image):
    """Scharr edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Scharr edge detection
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    
    # Combine
    edges = np.sqrt(scharr_x**2 + scharr_y**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return edges

def log_edge_detection(image, sigma=1.0):
    """Laplacian of Gaussian (LoG) edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Find zero crossings
    edges = np.zeros_like(laplacian)
    
    # Simple zero crossing detection
    for i in range(1, laplacian.shape[0]-1):
        for j in range(1, laplacian.shape[1]-1):
            if (laplacian[i,j] * laplacian[i+1,j] < 0 or
                laplacian[i,j] * laplacian[i,j+1] < 0 or
                laplacian[i,j] * laplacian[i-1,j] < 0 or
                laplacian[i,j] * laplacian[i,j-1] < 0):
                edges[i,j] = 255
    
    return edges.astype(np.uint8)

def morphological_edge_detection(image, operation='gradient'):
    """Morphological edge detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Define kernel
    kernel = np.ones((5,5), np.uint8)
    
    if operation == 'gradient':
        # Morphological gradient
        edges = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif operation == 'laplacian':
        # Morphological Laplacian
        dilated = cv2.dilate(gray, kernel, iterations=1)
        eroded = cv2.erode(gray, kernel, iterations=1)
        edges = dilated + eroded - 2 * gray
        edges = np.clip(edges, 0, 255)
    
    return edges

def edge_statistics(edges):
    """Calculate edge statistics"""
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    edge_density = edge_pixels / total_pixels
    
    # Edge intensity statistics
    edge_intensities = edges[edges > 0]
    if len(edge_intensities) > 0:
        mean_intensity = np.mean(edge_intensities)
        max_intensity = np.max(edge_intensities)
        std_intensity = np.std(edge_intensities)
    else:
        mean_intensity = max_intensity = std_intensity = 0
    
    return {
        'edge_pixels': edge_pixels,
        'total_pixels': total_pixels,
        'edge_density': edge_density,
        'mean_intensity': mean_intensity,
        'max_intensity': max_intensity,
        'std_intensity': std_intensity
    }

def edge_orientation_analysis(image):
    """Analyze edge orientations"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and orientation
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    
    # Filter by magnitude threshold
    threshold = np.mean(magnitude) + np.std(magnitude)
    significant_orientations = orientation[magnitude > threshold]
    
    return magnitude, orientation, significant_orientations

def run(image):
    """Run edge detection"""
    st.markdown("### 🔍 Edge Detection")
    
    # Edge detection method selection
    method = st.selectbox(
        "Edge Detection Method:",
        [
            "Canny", "Sobel", "Laplacian", "Prewitt", 
            "Roberts", "Scharr", "LoG", "Morphological"
        ]
    )
    
    # Method-specific parameters
    if method == "Canny":
        col1, col2, col3 = st.columns(3)
        with col1:
            low_threshold = st.slider("Low Threshold:", 1, 100, 50)
        with col2:
            high_threshold = st.slider("High Threshold:", 100, 300, 150)
        with col3:
            aperture_size = st.selectbox("Aperture Size:", [3, 5, 7])
        params = {'low_threshold': low_threshold, 'high_threshold': high_threshold, 'aperture_size': aperture_size}
        
    elif method == "Sobel":
        direction = st.selectbox("Direction:", ["both", "x", "y"])
        params = {'direction': direction}
        
    elif method == "Laplacian":
        kernel_size = st.selectbox("Kernel Size:", [1, 3, 5])
        params = {'kernel_size': kernel_size}
        
    elif method == "LoG":
        sigma = st.slider("Gaussian Sigma:", 0.5, 5.0, 1.0, 0.1)
        params = {'sigma': sigma}
        
    elif method == "Morphological":
        operation = st.selectbox("Operation:", ["gradient", "laplacian"])
        params = {'operation': operation}
        
    else:
        params = {}
    
    # Post-processing options
    st.subheader("🔧 Post-Processing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        apply_blur = st.checkbox("Apply Gaussian Blur")
        if apply_blur:
            blur_sigma = st.slider("Blur Sigma:", 0.5, 3.0, 1.0, 0.1)
    
    with col2:
        apply_threshold = st.checkbox("Apply Binary Threshold")
        if apply_threshold:
            threshold_value = st.slider("Threshold Value:", 0, 255, 127)
    
    if st.button("🚀 Detect Edges", type="primary"):
        with st.spinner(f"Applying {method} edge detection..."):
            
            try:
                # Apply edge detection based on method
                if method == "Canny":
                    edges = canny_edge_detection(image, **params)
                elif method == "Sobel":
                    edges = sobel_edge_detection(image, **params)
                elif method == "Laplacian":
                    edges = laplacian_edge_detection(image, **params)
                elif method == "Prewitt":
                    edges = prewitt_edge_detection(image)
                elif method == "Roberts":
                    edges = roberts_edge_detection(image)
                elif method == "Scharr":
                    edges = scharr_edge_detection(image)
                elif method == "LoG":
                    edges = log_edge_detection(image, **params)
                elif method == "Morphological":
                    edges = morphological_edge_detection(image, **params)
                
                # Apply post-processing
                if apply_blur:
                    edges = cv2.GaussianBlur(edges, (0, 0), blur_sigma)
                
                if apply_threshold:
                    _, edges = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY)
                
                # Display results
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Edges")
                    st.image(edges, caption=f"{method} Edges", use_column_width=True, clamp=True)
                
                with col3:
                    st.subheader("Overlay")
                    # Create overlay
                    img_array = np.array(image)
                    overlay = img_array.copy()
                    overlay[:, :, 0] = np.where(edges > 0, 255, overlay[:, :, 0])
                    st.image(overlay, caption="Edge Overlay", use_column_width=True)
                
                # Edge statistics
                st.subheader("📊 Edge Detection Statistics")
                
                stats = edge_statistics(edges)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Edge Pixels", f"{stats['edge_pixels']:,}")
                
                with col2:
                    st.metric("Edge Density", f"{stats['edge_density']*100:.2f}%")
                
                with col3:
                    st.metric("Mean Edge Intensity", f"{stats['mean_intensity']:.1f}")
                
                with col4:
                    st.metric("Max Edge Intensity", f"{stats['max_intensity']:.0f}")
                
                # Edge orientation analysis
                st.subheader("🧭 Edge Orientation Analysis")
                
                magnitude, orientation, significant_orientations = edge_orientation_analysis(image)
                
                if len(significant_orientations) > 0:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Edge magnitude
                    im1 = ax1.imshow(magnitude, cmap='hot')
                    ax1.set_title('Edge Magnitude')
                    ax1.axis('off')
                    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                    
                    # Edge orientation
                    im2 = ax2.imshow(orientation, cmap='hsv')
                    ax2.set_title('Edge Orientation')
                    ax2.axis('off')
                    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                    
                    # Orientation histogram
                    ax3.hist(significant_orientations, bins=36, alpha=0.7, color='#667eea')
                    ax3.set_xlabel('Orientation (degrees)')
                    ax3.set_ylabel('Frequency')
                    ax3.set_title('Edge Orientation Distribution')
                    ax3.grid(True, alpha=0.3)
                    
                    # Polar histogram
                    ax4 = plt.subplot(224, projection='polar')
                    theta = np.radians(significant_orientations)
                    ax4.hist(theta, bins=36, alpha=0.7, color='#764ba2')
                    ax4.set_title('Polar Orientation Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Dominant orientations
                    hist, bin_edges = np.histogram(significant_orientations, bins=36)
                    dominant_bins = np.argsort(hist)[-3:]  # Top 3 dominant orientations
                    
                    st.subheader("🎯 Dominant Edge Orientations")
                    for i, bin_idx in enumerate(dominant_bins[::-1]):
                        angle = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                        count = hist[bin_idx]
                        percentage = (count / len(significant_orientations)) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Rank {i+1}:**")
                        with col2:
                            st.write(f"Angle: {angle:.1f}°")
                        with col3:
                            st.write(f"Frequency: {percentage:.1f}%")
                
                # Edge contour analysis
                st.subheader("🔗 Edge Contour Analysis")
                
                # Find contours from edges
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Calculate contour properties
                    contour_areas = [cv2.contourArea(contour) for contour in contours]
                    contour_perimeters = [cv2.arcLength(contour, True) for contour in contours]
                    
                    # Filter significant contours
                    min_area = np.mean(contour_areas) if contour_areas else 0
                    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Contours", len(contours))
                    
                    with col2:
                        st.metric("Significant Contours", len(significant_contours))
                    
                    with col3:
                        avg_area = np.mean(contour_areas) if contour_areas else 0
                        st.metric("Average Area", f"{avg_area:.1f}")
                    
                    # Draw contours
                    if significant_contours:
                        img_with_contours = np.array(image).copy()
                        cv2.drawContours(img_with_contours, significant_contours, -1, (255, 0, 0), 2)
                        
                        st.subheader("🔴 Significant Contours")
                        st.image(img_with_contours, caption="Detected Contours", use_column_width=True)
                        
                        # Contour statistics
                        if len(contour_areas) > 1:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Area distribution
                            ax1.hist(contour_areas, bins=min(20, len(contour_areas)), alpha=0.7, color='#667eea')
                            ax1.set_xlabel('Contour Area')
                            ax1.set_ylabel('Frequency')
                            ax1.set_title('Contour Area Distribution')
                            ax1.grid(True, alpha=0.3)
                            
                            # Perimeter distribution
                            ax2.hist(contour_perimeters, bins=min(20, len(contour_perimeters)), alpha=0.7, color='#764ba2')
                            ax2.set_xlabel('Contour Perimeter')
                            ax2.set_ylabel('Frequency')
                            ax2.set_title('Contour Perimeter Distribution')
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                # Export options
                st.subheader("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Download Edge Map"):
                        import io
                        edge_img = Image.fromarray(edges)
                        img_bytes = io.BytesIO()
                        edge_img.save(img_bytes, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Edge Map",
                            data=img_bytes.getvalue(),
                            file_name=f"edges_{method.lower()}.png",
                            mime="image/png"
                        )
                
                with col2:
                    if st.button("Download Overlay"):
                        import io
                        overlay_img = Image.fromarray(overlay)
                        img_bytes = io.BytesIO()
                        overlay_img.save(img_bytes, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Overlay",
                            data=img_bytes.getvalue(),
                            file_name=f"edge_overlay_{method.lower()}.png",
                            mime="image/png"
                        )
                
                with col3:
                    if 'img_with_contours' in locals():
                        if st.button("Download Contours"):
                            import io
                            contour_img = Image.fromarray(img_with_contours)
                            img_bytes = io.BytesIO()
                            contour_img.save(img_bytes, format='PNG')
                            
                            st.download_button(
                                label="📥 Download Contours",
                                data=img_bytes.getvalue(),
                                file_name=f"contours_{method.lower()}.png",
                                mime="image/png"
                            )
                
            except Exception as e:
                st.error(f"Error during edge detection: {str(e)}")
                st.info("Please try with different parameters or a different image.")
