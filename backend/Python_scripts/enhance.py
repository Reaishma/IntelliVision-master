#!/usr/bin/env python3
"""
Image Enhancement Script for Laravel Backend
Processes images using CLAHE, gamma correction, unsharp masking, and denoising
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

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def apply_gamma_correction(image, gamma=1.2):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def apply_unsharp_masking(image, radius=1.0, amount=1.0):
    """Apply unsharp masking for sharpening"""
    gaussian = cv2.GaussianBlur(image, (0, 0), radius)
    unsharp_mask = cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)
    
    return unsharp_mask

def apply_denoising(image, h=10, template_window_size=7, search_window_size=21):
    """Apply Non-local Means Denoising"""
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                                  template_window_size, search_window_size)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, h, 
                                           template_window_size, search_window_size)
    
    return denoised

def apply_super_resolution(image, scale_factor=2):
    """Apply basic super resolution using interpolation"""
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Use INTER_CUBIC for upscaling
    super_res = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening after upscaling
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    super_res = cv2.filter2D(super_res, -1, kernel)
    
    return super_res

def image_to_base64(image):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def enhance_image(image_path, enhancement_type='clahe', parameters=None):
    """
    Enhance image using specified enhancement type
    """
    try:
        if parameters is None:
            parameters = {}
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        height, width = image.shape[:2]
        
        # Apply enhancement based on type
        if enhancement_type == 'clahe':
            clip_limit = parameters.get('clip_limit', 2.0)
            grid_size = parameters.get('grid_size', [8, 8])
            enhanced_image = apply_clahe(image, clip_limit, tuple(grid_size))
            
        elif enhancement_type == 'gamma':
            gamma = parameters.get('gamma', 1.2)
            enhanced_image = apply_gamma_correction(image, gamma)
            
        elif enhancement_type == 'unsharp':
            radius = parameters.get('radius', 1.0)
            amount = parameters.get('amount', 1.0)
            enhanced_image = apply_unsharp_masking(image, radius, amount)
            
        elif enhancement_type == 'denoise':
            h = parameters.get('h', 10)
            template_size = parameters.get('template_window_size', 7)
            search_size = parameters.get('search_window_size', 21)
            enhanced_image = apply_denoising(image, h, template_size, search_size)
            
        elif enhancement_type == 'super_resolution':
            scale_factor = parameters.get('scale_factor', 2)
            enhanced_image = apply_super_resolution(image, scale_factor)
            
        else:
            raise ValueError(f"Unknown enhancement type: {enhancement_type}")
        
        # Convert to base64
        enhanced_base64 = image_to_base64(enhanced_image)
        
        return {
            "enhanced_image": enhanced_base64,
            "enhancement_type": enhancement_type,
            "parameters": parameters,
            "original_dimensions": {"width": width, "height": height},
            "enhanced_dimensions": {"width": enhanced_image.shape[1], "height": enhanced_image.shape[0]}
        }
        
    except Exception as e:
        logger.error(f"Image enhancement error: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) < 4:
            raise ValueError("Usage: python enhance.py <image_path> <enhancement_type> <parameters_json>")
        
        image_path = sys.argv[1]
        enhancement_type = sys.argv[2]
        parameters_json = sys.argv[3]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Parse parameters
        try:
            parameters = json.loads(parameters_json)
        except json.JSONDecodeError:
            parameters = {}
        
        # Process enhancement
        result = enhance_image(image_path, enhancement_type, parameters)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "enhanced_image": "",
            "enhancement_type": "",
            "parameters": {}
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()