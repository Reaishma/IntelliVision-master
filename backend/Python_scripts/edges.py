#!/usr/bin/env python3
"""
Edge Detection Script for Laravel Backend
Processes images using Canny, Sobel, Laplacian, and Prewitt edge detectors
"""

import sys
import json
import numpy as np
import cv2
import os
from datetime import datetime
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_canny_edges(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """Apply Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)
    
    # Convert to 3-channel for consistency
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def apply_sobel_edges(image, ksize=3, dx=1, dy=1):
    """Apply Sobel edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel X
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize)
    sobel_x = np.uint8(np.absolute(sobel_x))
    
    # Sobel Y
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize)
    sobel_y = np.uint8(np.absolute(sobel_y))
    
    # Combined Sobel
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
    
    # Convert to 3-channel
    edges_bgr = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def apply_laplacian_edges(image, ksize=3):
    """Apply Laplacian edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Convert to 3-channel
    edges_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def apply_prewitt_edges(image):
    """Apply Prewitt edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float32)
    
    kernel_y = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=np.float32)
    
    # Apply kernels
    prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray, -1, kernel_y)
    
    # Combine
    prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt_combined = np.uint8(np.clip(prewitt_combined, 0, 255))
    
    # Convert to 3-channel
    edges_bgr = cv2.cvtColor(prewitt_combined, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def image_to_base64(image):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def detect_edges(image_path, method='canny', parameters=None):
    """
    Detect edges using specified method
    """
    try:
        if parameters is None:
            parameters = {}
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        height, width = image.shape[:2]
        
        # Apply edge detection based on method
        if method == 'canny':
            low_threshold = parameters.get('low_threshold', 50)
            high_threshold = parameters.get('high_threshold', 150)
            aperture_size = parameters.get('aperture_size', 3)
            edge_image = apply_canny_edges(image, low_threshold, high_threshold, aperture_size)
            
        elif method == 'sobel':
            ksize = parameters.get('ksize', 3)
            dx = parameters.get('dx', 1)
            dy = parameters.get('dy', 1)
            edge_image = apply_sobel_edges(image, ksize, dx, dy)
            
        elif method == 'laplacian':
            ksize = parameters.get('ksize', 3)
            edge_image = apply_laplacian_edges(image, ksize)
            
        elif method == 'prewitt':
            edge_image = apply_prewitt_edges(image)
            
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # Convert to base64
        edge_base64 = image_to_base64(edge_image)
        
        return {
            "edge_image": edge_base64,
            "method_used": method,
            "parameters": parameters,
            "image_dimensions": {"width": width, "height": height}
        }
        
    except Exception as e:
        logger.error(f"Edge detection error: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) < 4:
            raise ValueError("Usage: python edges.py <image_path> <method> <parameters_json>")
        
        image_path = sys.argv[1]
        method = sys.argv[2]
        parameters_json = sys.argv[3]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Parse parameters
        try:
            parameters = json.loads(parameters_json)
        except json.JSONDecodeError:
            parameters = {}
        
        # Process edge detection
        result = detect_edges(image_path, method, parameters)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "edge_image": "",
            "method_used": "",
            "parameters": {}
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()