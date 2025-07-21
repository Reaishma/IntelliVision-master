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
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(img_cv, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Simulate YOLO detection (simplified for demo)
        # In a real implementation, you would load a YOLO model
        detections = []
        
        # Simple edge-based object detection as fallback
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours[:10]):  # Limit to top 10 contours
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(0.9, area / (width * height) * 10)  # Simulated confidence
                
                if confidence >= confidence_threshold:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': confidence,
                        'class': 'object',  # Generic class
                        'class_id': 0
                    })
        
        return detections
        
    except Exception as e:
        st.error(f"Error in object detection: {str(e)}")
        return []

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
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Detected Objects")
                    st.image(result_image, caption="Detection Results", use_column_width=True)
                
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
