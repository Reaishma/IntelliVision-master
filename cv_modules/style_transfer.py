import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import matplotlib.pyplot as plt

def apply_artistic_filter(image, filter_type):
    """Apply artistic filters to simulate style transfer"""
    img_array = np.array(image)
    
    if filter_type == "Oil Painting":
        # Simulate oil painting effect
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter multiple times for oil painting effect
        for _ in range(3):
            img_cv = cv2.bilateralFilter(img_cv, 15, 80, 80)
        
        # Enhance edges
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine with filtered image
        result = cv2.bitwise_and(img_cv, edges)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
    elif filter_type == "Watercolor":
        # Simulate watercolor effect
        img_pil = image.copy()
        
        # Apply blur
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Enhance saturation
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(1.5)
        
        # Reduce number of colors
        img_pil = img_pil.quantize(colors=16).convert('RGB')
        
        result = np.array(img_pil)
        
    elif filter_type == "Pencil Sketch":
        # Create pencil sketch effect
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Invert the image
        gray_inv = 255 - gray
        
        # Apply Gaussian blur
        gray_inv_blur = cv2.GaussianBlur(gray_inv, (21, 21), 0)
        
        # Create sketch
        sketch = cv2.divide(gray, 255 - gray_inv_blur, scale=256)
        
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
    elif filter_type == "Cartoon":
        # Create cartoon effect
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter
        smooth = cv2.bilateralFilter(img_cv, 15, 80, 80)
        
        # Create edge mask
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine
        cartoon = cv2.bitwise_and(smooth, edges)
        result = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
        
    elif filter_type == "Pop Art":
        # Create pop art effect
        img_pil = image.copy()
        
        # Posterize (reduce colors)
        img_pil = img_pil.quantize(colors=8).convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(2.0)
        
        # Enhance saturation
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(2.0)
        
        result = np.array(img_pil)
        
    elif filter_type == "Vintage":
        # Create vintage effect
        img_pil = image.copy()
        
        # Apply sepia tone
        img_array = np.array(img_pil)
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        result = img_array @ sepia_filter.T
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Add vignette effect
        h, w = result.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = mask / mask.max()
        mask = 1 - mask * 0.3  # Adjust vignette strength
        mask = np.clip(mask, 0, 1)
        
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask
        
    else:
        result = img_array
    
    return Image.fromarray(result.astype(np.uint8))

def neural_style_transfer_simulation(content_image, style_type, strength=0.7):
    """Simulate neural style transfer with traditional image processing"""
    
    style_effects = {
        "Starry Night": {
            "blur_radius": 3,
            "color_enhance": 1.3,
            "contrast_enhance": 1.2,
            "swirl_strength": 0.1
        },
        "The Scream": {
            "blur_radius": 2,
            "color_enhance": 1.5,
            "contrast_enhance": 1.4,
            "saturation": 1.6
        },
        "Picasso": {
            "angular": True,
            "color_reduce": 12,
            "edge_enhance": True
        },
        "Monet": {
            "soft_blur": 4,
            "color_enhance": 1.1,
            "brightness": 1.1
        }
    }
    
    if style_type not in style_effects:
        return content_image
    
    effect = style_effects[style_type]
    img_pil = content_image.copy()
    
    # Apply style-specific effects
    if "blur_radius" in effect:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=effect["blur_radius"]))
    
    if "soft_blur" in effect:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=effect["soft_blur"]))
    
    if "color_enhance" in effect:
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(effect["color_enhance"])
    
    if "contrast_enhance" in effect:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(effect["contrast_enhance"])
    
    if "brightness" in effect:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(effect["brightness"])
    
    if "saturation" in effect:
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(effect["saturation"])
    
    if "color_reduce" in effect:
        img_pil = img_pil.quantize(colors=effect["color_reduce"]).convert('RGB')
    
    if "angular" in effect:
        # Create angular/cubist effect
        img_array = np.array(img_pil)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply edge detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine with original
        img_cv = cv2.bitwise_or(img_cv, edges)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Blend with original based on strength
    if strength < 1.0:
        img_array = np.array(img_pil)
        original_array = np.array(content_image)
        blended = img_array * strength + original_array * (1 - strength)
        img_pil = Image.fromarray(blended.astype(np.uint8))
    
    return img_pil

def run(image):
    """Run style transfer"""
    st.markdown("### 🎨 Style Transfer")
    
    # Style transfer method selection
    method = st.selectbox(
        "Style Transfer Method:",
        ["Artistic Filters", "Neural Style Transfer (Simulated)", "Custom Effects"]
    )
    
    if method == "Artistic Filters":
        style = st.selectbox(
            "Artistic Style:",
            ["Oil Painting", "Watercolor", "Pencil Sketch", "Cartoon", "Pop Art", "Vintage"]
        )
        
        if st.button("🎨 Apply Artistic Filter", type="primary"):
            with st.spinner(f"Applying {style} style..."):
                result_image = apply_artistic_filter(image, style)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader(f"{style} Style")
                    st.image(result_image, caption=f"{style} Result", use_column_width=True)
                
                # Download option
                st.subheader("💾 Export Result")
                import io
                img_bytes = io.BytesIO()
                result_image.save(img_bytes, format='PNG')
                
                st.download_button(
                    label=f"📥 Download {style}",
                    data=img_bytes.getvalue(),
                    file_name=f"{style.lower().replace(' ', '_')}_style.png",
                    mime="image/png"
                )
    
    elif method == "Neural Style Transfer (Simulated)":
        style = st.selectbox(
            "Famous Art Style:",
            ["Starry Night", "The Scream", "Picasso", "Monet"]
        )
        
        strength = st.slider(
            "Style Strength:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        if st.button("🧠 Apply Neural Style", type="primary"):
            with st.spinner(f"Applying {style} neural style transfer..."):
                result_image = neural_style_transfer_simulation(image, style, strength)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Content Image")
                    st.image(image, caption="Original", use_column_width=True)
                
                with col2:
                    st.subheader(f"{style} Style")
                    st.image(result_image, caption="Stylized Result", use_column_width=True)
                
                # Style analysis
                st.subheader("🔍 Style Analysis")
                
                # Calculate color distribution differences
                original_array = np.array(image)
                result_array = np.array(result_image)
                
                # Color statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    orig_brightness = np.mean(original_array)
                    result_brightness = np.mean(result_array)
                    brightness_change = ((result_brightness - orig_brightness) / orig_brightness) * 100
                    st.metric("Brightness Change", f"{brightness_change:+.1f}%")
                
                with col2:
                    orig_std = np.std(original_array)
                    result_std = np.std(result_array)
                    contrast_change = ((result_std - orig_std) / orig_std) * 100
                    st.metric("Contrast Change", f"{contrast_change:+.1f}%")
                
                with col3:
                    st.metric("Style Strength", f"{strength*100:.0f}%")
                
                # Color distribution comparison
                st.subheader("📊 Color Distribution Comparison")
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Original histograms
                for i, color in enumerate(['red', 'green', 'blue']):
                    ax1.hist(original_array[:,:,i].flatten(), bins=50, alpha=0.7, 
                            color=color, label=color.capitalize())
                ax1.set_title('Original Color Distribution')
                ax1.set_xlabel('Pixel Intensity')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                
                # Result histograms
                for i, color in enumerate(['red', 'green', 'blue']):
                    ax2.hist(result_array[:,:,i].flatten(), bins=50, alpha=0.7, 
                            color=color, label=color.capitalize())
                ax2.set_title('Stylized Color Distribution')
                ax2.set_xlabel('Pixel Intensity')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                
                # Difference image
                diff = np.abs(result_array.astype(float) - original_array.astype(float))
                ax3.imshow(diff.astype(np.uint8))
                ax3.set_title('Difference Map')
                ax3.axis('off')
                
                # Average color comparison
                orig_avg_color = np.mean(original_array, axis=(0,1))
                result_avg_color = np.mean(result_array, axis=(0,1))
                
                colors = ['Original', 'Stylized']
                red_vals = [orig_avg_color[0], result_avg_color[0]]
                green_vals = [orig_avg_color[1], result_avg_color[1]]
                blue_vals = [orig_avg_color[2], result_avg_color[2]]
                
                x = np.arange(len(colors))
                width = 0.25
                
                ax4.bar(x - width, red_vals, width, label='Red', color='red', alpha=0.7)
                ax4.bar(x, green_vals, width, label='Green', color='green', alpha=0.7)
                ax4.bar(x + width, blue_vals, width, label='Blue', color='blue', alpha=0.7)
                
                ax4.set_title('Average Color Comparison')
                ax4.set_ylabel('Average Intensity')
                ax4.set_xticks(x)
                ax4.set_xticklabels(colors)
                ax4.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download option
                st.subheader("💾 Export Result")
                import io
                img_bytes = io.BytesIO()
                result_image.save(img_bytes, format='PNG')
                
                st.download_button(
                    label=f"📥 Download {style} Style",
                    data=img_bytes.getvalue(),
                    file_name=f"neural_style_{style.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
    
    elif method == "Custom Effects":
        st.subheader("🎛️ Custom Style Controls")
        
        # Custom effect parameters
        col1, col2 = st.columns(2)
        
        with col1:
            blur_strength = st.slider("Blur Strength:", 0, 10, 0)
            brightness = st.slider("Brightness:", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast:", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            saturation = st.slider("Saturation:", 0.0, 2.0, 1.0, 0.1)
            color_quantize = st.slider("Color Levels:", 8, 256, 256)
            edge_enhance = st.checkbox("Edge Enhancement", False)
        
        if st.button("🎨 Apply Custom Effects", type="primary"):
            with st.spinner("Applying custom effects..."):
                result_image = image.copy()
                
                # Apply blur
                if blur_strength > 0:
                    result_image = result_image.filter(ImageFilter.GaussianBlur(radius=blur_strength))
                
                # Apply brightness
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(result_image)
                    result_image = enhancer.enhance(brightness)
                
                # Apply contrast
                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(result_image)
                    result_image = enhancer.enhance(contrast)
                
                # Apply saturation
                if saturation != 1.0:
                    enhancer = ImageEnhance.Color(result_image)
                    result_image = enhancer.enhance(saturation)
                
                # Apply color quantization
                if color_quantize < 256:
                    result_image = result_image.quantize(colors=color_quantize).convert('RGB')
                
                # Apply edge enhancement
                if edge_enhance:
                    result_image = result_image.filter(ImageFilter.EDGE_ENHANCE)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Custom Style")
                    st.image(result_image, caption="Custom Result", use_column_width=True)
                
                # Effect summary
                st.subheader("📋 Applied Effects Summary")
                effects_applied = []
                
                if blur_strength > 0:
                    effects_applied.append(f"Blur: {blur_strength}")
                if brightness != 1.0:
                    effects_applied.append(f"Brightness: {brightness:.1f}x")
                if contrast != 1.0:
                    effects_applied.append(f"Contrast: {contrast:.1f}x")
                if saturation != 1.0:
                    effects_applied.append(f"Saturation: {saturation:.1f}x")
                if color_quantize < 256:
                    effects_applied.append(f"Colors: {color_quantize} levels")
                if edge_enhance:
                    effects_applied.append("Edge Enhancement")
                
                if effects_applied:
                    for effect in effects_applied:
                        st.write(f"• {effect}")
                else:
                    st.write("No effects applied")
                
                # Download option
                st.subheader("💾 Export Result")
                import io
                img_bytes = io.BytesIO()
                result_image.save(img_bytes, format='PNG')
                
                st.download_button(
                    label="📥 Download Custom Style",
                    data=img_bytes.getvalue(),
                    file_name="custom_style.png",
                    mime="image/png"
                )
