#!/usr/bin/env python3
"""
Feature Detection Script for Laravel Backend
Processes images using Harris corners, SIFT, ORB, and FAST detectors
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

def detect_harris_corners(image):
    """Detect Harris corners"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Threshold for optimal value, may vary
    corners = cv2.dilate(corners, None)
    
    # Find corner coordinates
    corner_coords = np.where(corners > 0.01 * corners.max())
    features = []
    
    for y, x in zip(corner_coords[0], corner_coords[1]):
        features.append({
            "x": int(x),
            "y": int(y),
            "strength": float(corners[y, x])
        })
    
    # Draw corners on image
    result_image = image.copy()
    result_image[corners > 0.01 * corners.max()] = [0, 0, 255]
    
    return features, result_image

def detect_sift_features(image):
    """Detect SIFT features with ORB fallback"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        # Try SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        method_used = "SIFT"
    except AttributeError:
        # Fallback to ORB
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        method_used = "ORB (SIFT fallback)"
    
    features = []
    for kp in keypoints:
        features.append({
            "x": int(kp.pt[0]),
            "y": int(kp.pt[1]),
            "strength": float(kp.response)
        })
    
    # Draw keypoints
    result_image = cv2.drawKeypoints(image, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return features, result_image, method_used

def detect_orb_features(image):
    """Detect ORB features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    features = []
    for kp in keypoints:
        features.append({
            "x": int(kp.pt[0]),
            "y": int(kp.pt[1]),
            "strength": float(kp.response)
        })
    
    # Draw keypoints
    result_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    
    return features, result_image

def detect_fast_features(image):
    """Detect FAST features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    
    features = []
    for kp in keypoints:
        features.append({
            "x": int(kp.pt[0]),
            "y": int(kp.pt[1]),
            "strength": float(kp.response)
        })
    
    # Draw keypoints
    result_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))
    
    return features, result_image

def image_to_base64(image):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def detect_features(image_path, method='harris'):
    """
    Detect features using specified method
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        height, width = image.shape[:2]
        method_used = method
        
        # Apply feature detection method
        if method == 'harris':
            features, result_image = detect_harris_corners(image)
        elif method == 'sift':
            features, result_image, method_used = detect_sift_features(image)
        elif method == 'orb':
            features, result_image = detect_orb_features(image)
        elif method == 'fast':
            features, result_image = detect_fast_features(image)
        else:
            raise ValueError(f"Unknown feature detection method: {method}")
        
        # Limit features to top 100 by strength
        features.sort(key=lambda x: x['strength'], reverse=True)
        features = features[:100]
        
        # Convert result image to base64
        feature_image_base64 = image_to_base64(result_image)
        
        return {
            "features": features,
            "total_features": len(features),
            "method_used": method_used,
            "feature_image": feature_image_base64,
            "image_dimensions": {"width": width, "height": height}
        }
        
    except Exception as e:
        logger.error(f"Feature detection error: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python features.py <image_path> <method>")
        
        image_path = sys.argv[1]
        method = sys.argv[2]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Process feature detection
        result = detect_features(image_path, method)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "features": [],
            "total_features": 0,
            "feature_image": ""
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()