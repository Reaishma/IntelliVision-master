import streamlit as st
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def kmeans_segmentation(image, n_clusters=5):
    """K-means clustering for image segmentation"""
    img_array = np.array(image)
    
    # Reshape image for clustering
    pixel_values = img_array.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    
    # Map back to image shape
    segmented = labels.reshape(img_array.shape[:2])
    
    return segmented, kmeans.cluster_centers_

def slic_segmentation(image, n_segments=100, compactness=10):
    """SLIC superpixel segmentation"""
    img_array = np.array(image)
    segments = slic(img_array, n_segments=n_segments, compactness=compactness, start_label=1)
    return segments

def watershed_segmentation(image):
    """Watershed segmentation"""
    img_array = np.array(image)
    gray = rgb2gray(img_array)
    
    # Find local maxima
    elevation_map = sobel(gray)
    
    # Find markers
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[gray < 0.3] = 1
    markers[gray > 0.7] = 2
    
    # Apply watershed
    segmentation = watershed(elevation_map, markers)
    
    return segmentation

def felzenszwalb_segmentation(image, scale=100, sigma=0.5, min_size=50):
    """Felzenszwalb's efficient graph-based segmentation"""
    img_array = np.array(image)
    segments = felzenszwalb(img_array, scale=scale, sigma=sigma, min_size=min_size)
    return segments

def quickshift_segmentation(image, kernel_size=3, max_dist=6, ratio=0.5):
    """Quick shift segmentation"""
    img_array = np.array(image)
    segments = quickshift(img_array, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    return segments

def create_segmentation_overlay(image, segments, alpha=0.6):
    """Create overlay of segmentation on original image"""
    img_array = np.array(image)
    
    # Create colored segmentation map
    segmented_colored = np.zeros_like(img_array)
    unique_segments = np.unique(segments)
    
    # Generate colors for each segment
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_segments)))[:, :3] * 255
    
    for i, segment_id in enumerate(unique_segments):
        mask = segments == segment_id
        segmented_colored[mask] = colors[i % len(colors)]
    
    # Create overlay
    overlay = img_array * (1 - alpha) + segmented_colored * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay), Image.fromarray(segmented_colored.astype(np.uint8))

def run(image):
    """Run image segmentation"""
    st.markdown("### 🎭 Image Segmentation")
    
    # Segmentation method selection
    method = st.selectbox(
        "Segmentation Method:",
        ["K-Means Clustering", "SLIC Superpixels", "Watershed", "Felzenszwalb", "Quick Shift"]
    )
    
    # Method-specific parameters
    if method == "K-Means Clustering":
        n_clusters = st.slider("Number of Clusters:", 2, 10, 5)
        params = {'n_clusters': n_clusters}
        
    elif method == "SLIC Superpixels":
        col1, col2 = st.columns(2)
        with col1:
            n_segments = st.slider("Number of Segments:", 50, 500, 100)
        with col2:
            compactness = st.slider("Compactness:", 1, 50, 10)
        params = {'n_segments': n_segments, 'compactness': compactness}
        
    elif method == "Watershed":
        params = {}
        
    elif method == "Felzenszwalb":
        col1, col2, col3 = st.columns(3)
        with col1:
            scale = st.slider("Scale:", 50, 300, 100)
        with col2:
            sigma = st.slider("Sigma:", 0.1, 2.0, 0.5)
        with col3:
            min_size = st.slider("Min Size:", 10, 100, 50)
        params = {'scale': scale, 'sigma': sigma, 'min_size': min_size}
        
    elif method == "Quick Shift":
        col1, col2, col3 = st.columns(3)
        with col1:
            kernel_size = st.slider("Kernel Size:", 1, 10, 3)
        with col2:
            max_dist = st.slider("Max Distance:", 1, 20, 6)
        with col3:
            ratio = st.slider("Ratio:", 0.1, 1.0, 0.5)
        params = {'kernel_size': kernel_size, 'max_dist': max_dist, 'ratio': ratio}
    
    # Overlay transparency
    alpha = st.slider("Overlay Transparency:", 0.0, 1.0, 0.6, 0.1)
    
    if st.button("🎨 Segment Image", type="primary"):
        with st.spinner(f"Applying {method} segmentation..."):
            
            try:
                # Apply segmentation based on method
                if method == "K-Means Clustering":
                    segments, centers = kmeans_segmentation(image, **params)
                    
                elif method == "SLIC Superpixels":
                    segments = slic_segmentation(image, **params)
                    
                elif method == "Watershed":
                    segments = watershed_segmentation(image)
                    
                elif method == "Felzenszwalb":
                    segments = felzenszwalb_segmentation(image, **params)
                    
                elif method == "Quick Shift":
                    segments = quickshift_segmentation(image, **params)
                
                # Create visualizations
                overlay_img, segmented_img = create_segmentation_overlay(image, segments, alpha)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_container_width=True)
                
                with col2:
                    st.subheader("Segmentation")
                    st.image(segmented_img, caption="Segmented Image", use_container_width=True)
                
                with col3:
                    st.subheader("Overlay")
                    st.image(overlay_img, caption="Segmentation Overlay", use_container_width=True)
                
                # Segmentation statistics
                st.subheader("📊 Segmentation Statistics")
                
                unique_segments = np.unique(segments)
                num_segments = len(unique_segments)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Segments", num_segments)
                
                with col2:
                    avg_segment_size = segments.size / num_segments
                    st.metric("Avg Segment Size", f"{avg_segment_size:.0f} pixels")
                
                with col3:
                    largest_segment = np.max(np.bincount(segments.flatten()))
                    st.metric("Largest Segment", f"{largest_segment} pixels")
                
                with col4:
                    smallest_segment = np.min(np.bincount(segments.flatten()))
                    st.metric("Smallest Segment", f"{smallest_segment} pixels")
                
                # Segment size distribution
                st.subheader("📈 Segment Size Distribution")
                
                segment_sizes = [np.sum(segments == seg_id) for seg_id in unique_segments]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histogram
                ax1.hist(segment_sizes, bins=min(20, num_segments), alpha=0.7, color='#667eea')
                ax1.set_xlabel('Segment Size (pixels)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Segment Size Distribution')
                ax1.grid(True, alpha=0.3)
                
                # Box plot
                ax2.boxplot(segment_sizes, patch_artist=True, 
                           boxprops=dict(facecolor='#667eea', alpha=0.7))
                ax2.set_ylabel('Segment Size (pixels)')
                ax2.set_title('Segment Size Statistics')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # K-means specific info
                if method == "K-Means Clustering":
                    st.subheader("🎨 Color Cluster Centers")
                    
                    # Display cluster colors
                    cluster_cols = st.columns(min(len(centers), 5))
                    for i, center in enumerate(centers[:5]):
                        with cluster_cols[i]:
                            color_rgb = center.astype(int)
                            color_hex = "#{:02x}{:02x}{:02x}".format(color_rgb[0], color_rgb[1], color_rgb[2])
                            st.color_picker(f"Cluster {i+1}", color_hex, disabled=True)
                            st.write(f"RGB: {color_rgb}")
                
                # Export options
                st.subheader("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Download Segmentation"):
                        import io
                        img_bytes = io.BytesIO()
                        segmented_img.save(img_bytes, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Segmentation",
                            data=img_bytes.getvalue(),
                            file_name=f"segmentation_{method.lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )
                
                with col2:
                    if st.button("Download Overlay"):
                        import io
                        img_bytes = io.BytesIO()
                        overlay_img.save(img_bytes, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Overlay",
                            data=img_bytes.getvalue(),
                            file_name=f"overlay_{method.lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )
                
            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")
                st.info("Please try with different parameters or a different image.")
