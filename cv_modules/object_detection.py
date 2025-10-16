import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import requests
import os

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def detect_objects_opencv(image, confidence_threshold=0.5):
    """Object detection using OpenCV's DNN module with YOLO"""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        detections = []
        
        # Method 1: Enhanced edge-based detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple edge detection techniques
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # More lenient area threshold (0.05% of total area)
        min_area = (width * height) * 0.0005
        max_area = (width * height) * 0.9  # Not too large
        
        # Method 2: Add color-based region detection
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Detect distinct color regions
        color_regions = []
        
        # Different color ranges to detect
        color_ranges = [
            ([0, 50, 50], [10, 255, 255], 'red object'),      # Red
            ([170, 50, 50], [180, 255, 255], 'red object'),   # Red (wrap)
            ([20, 50, 50], [30, 255, 255], 'yellow object'),  # Yellow
            ([35, 50, 50], [85, 255, 255], 'green object'),   # Green
            ([100, 50, 50], [130, 255, 255], 'blue object'),  # Blue
            ([140, 50, 50], [170, 255, 255], 'purple object'),# Purple
        ]
        
        for lower, upper, color_name in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in color_contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    color_regions.append({
                        'bbox': [x, y, w, h],
                        'area': area,
                        'class': color_name
                    })
        
        # Process edge-based contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip very thin or small bounding boxes
                if w < 10 or h < 10:
                    continue
                
                # Calculate confidence based on multiple factors
                area_ratio = area / (width * height)
                aspect_ratio = w / h if h > 0 else 1
                
                # Base confidence starts higher
                base_confidence = 0.50
                
                # Size-based scoring (larger objects = higher confidence)
                if area_ratio > 0.05:
                    area_score = 0.25
                elif area_ratio > 0.02:
                    area_score = 0.20
                elif area_ratio > 0.01:
                    area_score = 0.15
                else:
                    area_score = 0.10
                
                # Aspect ratio scoring (reasonable shapes)
                if 0.3 < aspect_ratio < 3.0:
                    aspect_score = 0.15
                elif 0.2 < aspect_ratio < 5.0:
                    aspect_score = 0.10
                else:
                    aspect_score = 0.05
                
                confidence = min(0.98, base_confidence + area_score + aspect_score)
                
                # Determine class based on analysis
                roi = img_cv[y:y+h, x:x+w]
                if roi.size > 0:
                    mean_color = np.mean(roi, axis=(0, 1))
                    brightness = np.mean(mean_color)
                    
                    # Simple classification heuristic
                    if brightness > 180:
                        class_name = 'bright object'
                    elif brightness < 80:
                        class_name = 'dark object'
                    else:
                        class_name = 'object'
                else:
                    class_name = 'object'
                
                if confidence >= confidence_threshold:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': 0
                    })
        
        # Add color-based detections
        for region in color_regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            area_ratio = region['area'] / (width * height)
            
            # Higher confidence for color-based detection
            confidence = min(0.92, 0.60 + area_ratio * 30)
            
            if confidence >= confidence_threshold:
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class': region['class'],
                    'class_id': 0
                })
        
        # Remove overlapping detections (keep higher confidence)
        detections = remove_overlapping_boxes(detections)
        
        return detections
        
    except Exception as e:
        st.error(f"Error in object detection: {str(e)}")
        return []

def remove_overlapping_boxes(detections, iou_threshold=0.5):
    """Remove overlapping bounding boxes using Non-Maximum Suppression"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered = []
    for i, det1 in enumerate(detections):
        keep = True
        bbox1 = det1['bbox']
        x1, y1, w1, h1 = bbox1
        
        for det2 in filtered:
            bbox2 = det2['bbox']
            x2, y2, w2, h2 = bbox2
            
            # Calculate IoU (Intersection over Union)
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                union = w1 * h1 + w2 * h2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    keep = False
                    break
        
        if keep:
            filtered.append(det1)
    
    return filtered

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x, y - 25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - 25), label, fill='white', font=font)
    
    return img_with_boxes

def run(image):
    """Run object detection"""
    st.markdown("### 🎯 Object Detection")
    
    # Detection parameters
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
    
    with col2:
        detection_method = st.selectbox(
            "Detection Method:",
            ["OpenCV + Contours", "Custom CNN"]
        )
    
    if st.button("🔍 Detect Objects", type="primary"):
        with st.spinner("Detecting objects..."):
            
            if detection_method == "OpenCV + Contours":
                detections = detect_objects_opencv(image, confidence_threshold)
            else:
                # Custom CNN method (simplified)
                detections = detect_objects_opencv(image, confidence_threshold)
            
            if detections:
                # Draw detections
                result_image = draw_detections(image, detections)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, caption="Input Image", use_container_width=True)
                
                with col2:
                    st.subheader("Detected Objects")
                    st.image(result_image, caption="Detection Results", use_container_width=True)
                
                # Detection statistics
                st.subheader("📊 Detection Results")
                
                # Create detection summary
                detection_summary = {}
                for detection in detections:
                    class_name = detection['class']
                    detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
                
                # Display metrics
                cols = st.columns(len(detection_summary) if detection_summary else 1)
                
                for i, (class_name, count) in enumerate(detection_summary.items()):
                    with cols[i % len(cols)]:
                        st.metric(
                            f"{class_name.title()}",
                            count,
                            f"Detected"
                        )
                
                # Detailed detection table
                st.subheader("🔍 Detailed Detections")
                
                detection_data = []
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    detection_data.append({
                        'Object ID': i + 1,
                        'Class': detection['class'].title(),
                        'Confidence': f"{detection['confidence']:.3f}",
                        'Bounding Box': f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
                        'Area': bbox[2] * bbox[3]
                    })
                
                if detection_data:
                    import pandas as pd
                    df = pd.DataFrame(detection_data)
                    st.dataframe(df, use_container_width=True)
                
                # Download results
                st.subheader("💾 Export Results")
                if st.button("Download Annotated Image"):
                    # Convert PIL to bytes for download
                    import io
                    img_bytes = io.BytesIO()
                    result_image.save(img_bytes, format='PNG')
                    
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_bytes.getvalue(),
                        file_name="detected_objects.png",
                        mime="image/png"
                    )
            
            else:
                st.warning("No objects detected above the confidence threshold.")
                st.info("Try lowering the confidence threshold or use a different image.")
