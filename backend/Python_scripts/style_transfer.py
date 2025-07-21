#!/usr/bin/env python3
"""
Style Transfer Script for Laravel Backend
Processes images using artistic filters and effects
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

def apply_oil_painting(image):
    """Apply oil painting effect"""
    # Use OpenCV's oil painting effect
    oil_painting = cv2.xphoto.oilPainting(image, 7, 1)
    return oil_painting

def apply_watercolor(image):
    """Apply watercolor effect"""
    # Create watercolor effect using bilateral filter and edge detection
    blur = cv2.bilateralFilter(image, 15, 80, 80)
    
    # Edge detection
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine with blurred image
    watercolor = cv2.bitwise_and(blur, edges)
    return watercolor

def apply_pencil_sketch(image):
    """Apply pencil sketch effect"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the image
    gray_inv = 255 - gray
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray_inv, (21, 21), 0)
    
    # Blend the images
    blend = cv2.divide(gray, 255 - blur, scale=256)
    
    # Convert back to BGR
    sketch = cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
    return sketch

def apply_cartoon(image):
    """Apply cartoon effect"""
    # Bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(image, 15, 80, 80)
    
    # Edge detection
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # Convert edges to 3-channel
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine bilateral filter with edges
    cartoon = cv2.bitwise_and(bilateral, edges)
    return cartoon

def apply_vintage(image):
    """Apply vintage effect"""
    # Create vintage effect with color adjustments
    vintage = image.copy().astype(np.float32)
    
    # Apply sepia tone
    kernel = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
    
    vintage = cv2.transform(vintage, kernel)
    vintage = np.clip(vintage, 0, 255)
    
    # Add noise and adjust brightness/contrast
    noise = np.random.normal(0, 15, vintage.shape)
    vintage = vintage + noise
    vintage = np.clip(vintage, 0, 255)
    
    # Reduce saturation and add warmth
    vintage = vintage * 0.9 + 25
    vintage = np.clip(vintage, 0, 255).astype(np.uint8)
    
    return vintage

def image_to_base64(image):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def apply_style_transfer(image_path, style='oil_painting'):
    """
    Apply style transfer using specified style
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        height, width = image.shape[:2]
        
        # Apply style based on method
        if style == 'oil_painting':
            try:
                styled_image = apply_oil_painting(image)
            except:
                # Fallback if xphoto not available
                styled_image = cv2.bilateralFilter(image, 15, 80, 80)
        elif style == 'watercolor':
            styled_image = apply_watercolor(image)
        elif style == 'pencil_sketch':
            styled_image = apply_pencil_sketch(image)
        elif style == 'cartoon':
            styled_image = apply_cartoon(image)
        elif style == 'vintage':
            styled_image = apply_vintage(image)
        else:
            raise ValueError(f"Unknown style: {style}")
        
        # Convert to base64
        styled_base64 = image_to_base64(styled_image)
        
        return {
            "styled_image": styled_base64,
            "style_applied": style,
            "image_dimensions": {"width": width, "height": height}
        }
        
    except Exception as e:
        logger.error(f"Style transfer error: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python style_transfer.py <image_path> <style>")
        
        image_path = sys.argv[1]
        style = sys.argv[2]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Process style transfer
        result = apply_style_transfer(image_path, style)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "styled_image": "",
            "style_applied": ""
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()