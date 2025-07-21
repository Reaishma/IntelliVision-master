#!/usr/bin/env python3
"""
Image Segmentation Script for Laravel Backend
Processes images using K-means, SLIC, and watershed segmentation
"""

import sys
import json
import numpy as np
import cv2
import os
from datetime import datetime
import logging
import base64
from sklearn.cluster import KMeans
from skimage.segmentation import slic, watershed
from skimage.feature import peak_local_maxima
from skimage.filters import sobel
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def segment_kmeans(image, clusters=3):
    """Segment image using K-means clustering"""
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)
    
    return segmented_image, clusters

def segment_slic(image, n_segments=100):
    """Segment image using SLIC superpixels"""
    segments = slic(image, n_segments=n_segments, compactness=10, sigma=1, start_label=1)
    
    # Create colored segmentation
    segmented_image = np.zeros_like(image)
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        segmented_image[mask] = np.mean(image[mask], axis=0)
    
    return segmented_image.astype(np.uint8), len(np.unique(segments))

def segment_watershed(image, markers=10):
    """Segment image using watershed algorithm"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find local maxima
    local_maxima = peak_local_maxima(gray, min_distance=20, threshold_abs=30, indices=False)
    markers_img = ndimage.label(local_maxima)[0]
    
    # Apply watershed
    gradient = sobel(gray)
    segments = watershed(gradient, markers_img)
    
    # Create colored segmentation
    segmented_image = np.zeros_like(image)
    for segment_id in np.unique(segments):
        if segment_id == 0:  # Skip background
            continue
        mask = segments == segment_id
        segmented_image[mask] = np.mean(image[mask], axis=0)
    
    return segmented_image.astype(np.uint8), len(np.unique(segments)) - 1

def image_to_base64(image):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def segment_image(image_path, method='kmeans', clusters=3):
    """
    Segment image using specified method
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Apply segmentation method
        if method == 'kmeans':
            segmented_img, segments_count = segment_kmeans(image_rgb, clusters)
        elif method == 'slic':
            segmented_img, segments_count = segment_slic(image_rgb, clusters * 30)
        elif method == 'watershed':
            segmented_img, segments_count = segment_watershed(image_rgb, clusters * 3)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Convert to base64
        segmented_base64 = image_to_base64(segmented_img)
        
        return {
            "segmented_image": segmented_base64,
            "segments_count": segments_count,
            "method_used": method,
            "image_dimensions": {"width": width, "height": height}
        }
        
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) < 4:
            raise ValueError("Usage: python segment.py <image_path> <method> <clusters>")
        
        image_path = sys.argv[1]
        method = sys.argv[2]
        clusters = int(sys.argv[3])
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Process segmentation
        result = segment_image(image_path, method, clusters)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "segmented_image": "",
            "segments_count": 0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()