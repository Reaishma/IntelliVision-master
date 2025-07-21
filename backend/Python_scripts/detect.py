#!/usr/bin/env python3
"""
Object Detection Script for Laravel Backend
Processes images using OpenCV contour detection
"""

import sys
import json
import numpy as np
import cv2
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_objects_contour(image_path, confidence_threshold=0.5):
    """
    Detect objects using contour detection
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        
        for i, contour in enumerate(contours):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area threshold
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            aspect_ratio = float(w) / h
            
            # Determine object class based on shape characteristics
            if circularity > 0.7:
                object_class = "circular_object"
                confidence = min(0.9, circularity)
            elif aspect_ratio > 3 or aspect_ratio < 0.33:
                object_class = "elongated_object"
                confidence = min(0.8, 1.0 / abs(aspect_ratio - 1) if abs(aspect_ratio - 1) > 0 else 0.8)
            elif 0.8 < aspect_ratio < 1.2:
                object_class = "square_object"
                confidence = min(0.85, 1.0 - abs(aspect_ratio - 1))
            else:
                object_class = "rectangular_object"
                confidence = 0.7
            
            # Apply confidence threshold
            if confidence >= confidence_threshold:
                objects.append({
                    "class": object_class,
                    "confidence": round(confidence, 3),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(area),
                    "aspect_ratio": round(aspect_ratio, 2)
                })
        
        return {
            "objects": objects,
            "total_objects": len(objects),
            "image_dimensions": {"width": width, "height": height}
        }
        
    except Exception as e:
        logger.error(f"Object detection error: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python detect.py <image_path> <confidence_threshold>")
        
        image_path = sys.argv[1]
        confidence_threshold = float(sys.argv[2])
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Process object detection
        result = detect_objects_contour(image_path, confidence_threshold)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "objects": [],
            "total_objects": 0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()