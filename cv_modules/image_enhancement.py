import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, restoration, filters

def histogram_equalization(image):
    """Apply histogram equalization"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        # Convert to YUV and equalize Y channel
        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        result = cv2.equalizeHist(img_array)
    
    return Image.fromarray(result)

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply(img_array)
    
    return Image.fromarray(result)

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction"""
    img_array = np.array(image).astype(np.float32) / 255.0
    corrected = np.power(img_array, gamma)
    result = (corrected * 255).astype(np.uint8)
    return Image.fromarray(result)

def unsharp_masking(image, radius=1.0, amount=1.0):
    """Apply unsharp masking for sharpening"""
    img_array = np.array(image)
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
    
    # Create unsharp mask
    sharpened = cv2.addWeighted(img_array, 1.0 + amount, blurred, -amount, 0)
    
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

def noise_reduction(image, method="bilateral"):
    """Apply noise reduction"""
    img_array = np.array(image)
    
    if method == "bilateral":
        # Bilateral filtering
        result = cv2.bilateralFilter(img_array, 9, 75, 75)
    elif method == "gaussian":
        # Gaussian blur
        result = cv2.GaussianBlur(img_array, (5, 5), 0)
    elif method == "median":
        # Median filtering
        result = cv2.medianBlur(img_array, 5)
    elif method == "non_local_means":
        # Non-local means denoising
        if len(img_array.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        else:
            result = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
    else:
        result = img_array
    
    return Image.fromarray(result)

def white_balance_correction(image, method="gray_world"):
    """Apply white balance correction"""
    img_array = np.array(image).astype(np.float32)
    
    if method == "gray_world":
        # Gray world assumption
        mean_r = np.mean(img_array[:,:,0])
        mean_g = np.mean(img_array[:,:,1])
        mean_b = np.mean(img_array[:,:,2])
        
        # Calculate scaling factors
        mean_gray = (mean_r + mean_g + mean_b) / 3
        scale_r = mean_gray / mean_r
        scale_g = mean_gray / mean_g
        scale_b = mean_gray / mean_b
        
        # Apply scaling
        result = img_array.copy()
        result[:,:,0] *= scale_r
        result[:,:,1] *= scale_g
        result[:,:,2] *= scale_b
        
    elif method == "white_patch":
        # White patch assumption
        max_r = np.max(img_array[:,:,0])
        max_g = np.max(img_array[:,:,1])
        max_b = np.max(img_array[:,:,2])
        
        # Calculate scaling factors
        scale_r = 255 / max_r
        scale_g = 255 / max_g
        scale_b = 255 / max_b
        
        # Apply scaling
        result = img_array.copy()
        result[:,:,0] *= scale_r
        result[:,:,1] *= scale_g
        result[:,:,2] *= scale_b
    else:
        result = img_array
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)

def color_enhancement(image, saturation=1.0, vibrance=0.0):
    """Enhance colors with saturation and vibrance"""
    img_pil = image.copy()
    
    # Apply saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(saturation)
    
    # Apply vibrance (selective saturation)
    if vibrance != 0.0:
        img_array = np.array(img_pil).astype(np.float32)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate saturation mask (less saturated pixels get more enhancement)
        sat_mask = 1 - (hsv[:,:,1] / 255.0)
        
        # Apply vibrance
        hsv[:,:,1] = hsv[:,:,1] + (vibrance * sat_mask * 50)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        img_pil = Image.fromarray(result)
    
    return img_pil

def exposure_adjustment(image, exposure_value=0.0):
    """Adjust exposure"""
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply exposure adjustment
    exposed = img_array * (2.0 ** exposure_value)
    exposed = np.clip(exposed, 0, 1)
    
    result = (exposed * 255).astype(np.uint8)
    return Image.fromarray(result)

def run(image):
    """Run image enhancement"""
    st.markdown("### ✨ Image Enhancement")
    
    # Enhancement method selection
    method = st.selectbox(
        "Enhancement Method:",
        [
            "Contrast Enhancement",
            "Brightness & Exposure",
            "Color Enhancement", 
            "Sharpening",
            "Noise Reduction",
            "White Balance",
            "Custom Enhancement"
        ]
    )
    
    if method == "Contrast Enhancement":
        st.subheader("📊 Contrast Enhancement")
        
        contrast_method = st.selectbox(
            "Contrast Method:",
            ["Histogram Equalization", "CLAHE", "Gamma Correction"]
        )
        
        if contrast_method == "CLAHE":
            col1, col2 = st.columns(2)
            with col1:
                clip_limit = st.slider("Clip Limit:", 1.0, 10.0, 2.0, 0.5)
            with col2:
                tile_size = st.slider("Tile Size:", 4, 16, 8)
            params = {'clip_limit': clip_limit, 'tile_grid_size': (tile_size, tile_size)}
        elif contrast_method == "Gamma Correction":
            gamma = st.slider("Gamma:", 0.1, 3.0, 1.0, 0.1)
            params = {'gamma': gamma}
        else:
            params = {}
        
        if st.button("🚀 Enhance Contrast", type="primary"):
            with st.spinner(f"Applying {contrast_method}..."):
                if contrast_method == "Histogram Equalization":
                    result_image = histogram_equalization(image)
                elif contrast_method == "CLAHE":
                    result_image = clahe_enhancement(image, **params)
                elif contrast_method == "Gamma Correction":
                    result_image = gamma_correction(image, **params)
                
                # Display results with histograms
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                    
                    # Original histogram
                    fig, ax = plt.subplots(figsize=(6, 3))
                    img_array = np.array(image)
                    for i, color in enumerate(['red', 'green', 'blue']):
                        ax.hist(img_array[:,:,i].flatten(), bins=50, alpha=0.7, 
                               color=color, label=color.capitalize())
                    ax.set_title('Original Histogram')
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Enhanced")
                    st.image(result_image, caption=f"{contrast_method} Result", use_column_width=True)
                    
                    # Enhanced histogram
                    fig, ax = plt.subplots(figsize=(6, 3))
                    result_array = np.array(result_image)
                    for i, color in enumerate(['red', 'green', 'blue']):
                        ax.hist(result_array[:,:,i].flatten(), bins=50, alpha=0.7, 
                               color=color, label=color.capitalize())
                    ax.set_title('Enhanced Histogram')
                    ax.set_xlabel('Pixel Intensity')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    st.pyplot(fig)
                
                # Metrics
                st.subheader("📈 Enhancement Metrics")
                
                orig_contrast = np.std(np.array(image))
                enhanced_contrast = np.std(np.array(result_image))
                contrast_improvement = ((enhanced_contrast - orig_contrast) / orig_contrast) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Contrast", f"{orig_contrast:.2f}")
                with col2:
                    st.metric("Enhanced Contrast", f"{enhanced_contrast:.2f}")
                with col3:
                    st.metric("Improvement", f"{contrast_improvement:+.1f}%")
    
    elif method == "Brightness & Exposure":
        st.subheader("☀️ Brightness & Exposure")
        
        col1, col2 = st.columns(2)
        with col1:
            brightness = st.slider("Brightness:", 0.5, 2.0, 1.0, 0.1)
        with col2:
            exposure = st.slider("Exposure:", -2.0, 2.0, 0.0, 0.1)
        
        if st.button("🌟 Adjust Brightness & Exposure", type="primary"):
            with st.spinner("Adjusting brightness and exposure..."):
                # Apply brightness
                enhancer = ImageEnhance.Brightness(image)
                bright_image = enhancer.enhance(brightness)
                
                # Apply exposure
                result_image = exposure_adjustment(bright_image, exposure)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Adjusted")
                    st.image(result_image, caption="Brightness & Exposure Adjusted", use_column_width=True)
                
                # Brightness analysis
                st.subheader("📊 Brightness Analysis")
                
                orig_brightness = np.mean(np.array(image))
                result_brightness = np.mean(np.array(result_image))
                brightness_change = ((result_brightness - orig_brightness) / orig_brightness) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Brightness", f"{orig_brightness:.1f}")
                with col2:
                    st.metric("Adjusted Brightness", f"{result_brightness:.1f}")
                with col3:
                    st.metric("Change", f"{brightness_change:+.1f}%")
    
    elif method == "Color Enhancement":
        st.subheader("🌈 Color Enhancement")
        
        col1, col2 = st.columns(2)
        with col1:
            saturation = st.slider("Saturation:", 0.0, 2.0, 1.0, 0.1)
        with col2:
            vibrance = st.slider("Vibrance:", -1.0, 1.0, 0.0, 0.1)
        
        if st.button("🎨 Enhance Colors", type="primary"):
            with st.spinner("Enhancing colors..."):
                result_image = color_enhancement(image, saturation, vibrance)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Enhanced")
                    st.image(result_image, caption="Color Enhanced", use_column_width=True)
                
                # Color analysis
                st.subheader("🎯 Color Analysis")
                
                # Calculate color statistics
                orig_array = np.array(image)
                result_array = np.array(result_image)
                
                # Convert to HSV for saturation analysis
                orig_hsv = cv2.cvtColor(orig_array, cv2.COLOR_RGB2HSV)
                result_hsv = cv2.cvtColor(result_array, cv2.COLOR_RGB2HSV)
                
                orig_sat = np.mean(orig_hsv[:,:,1])
                result_sat = np.mean(result_hsv[:,:,1])
                sat_change = ((result_sat - orig_sat) / orig_sat) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Saturation", f"{orig_sat:.1f}")
                with col2:
                    st.metric("Enhanced Saturation", f"{result_sat:.1f}")
                with col3:
                    st.metric("Change", f"{sat_change:+.1f}%")
    
    elif method == "Sharpening":
        st.subheader("🔍 Image Sharpening")
        
        col1, col2 = st.columns(2)
        with col1:
            radius = st.slider("Sharpening Radius:", 0.5, 5.0, 1.0, 0.1)
        with col2:
            amount = st.slider("Sharpening Amount:", 0.5, 3.0, 1.0, 0.1)
        
        if st.button("⚡ Sharpen Image", type="primary"):
            with st.spinner("Sharpening image..."):
                result_image = unsharp_masking(image, radius, amount)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Sharpened")
                    st.image(result_image, caption="Sharpened Image", use_column_width=True)
                
                # Sharpness analysis
                st.subheader("📏 Sharpness Analysis")
                
                # Calculate edge strength as sharpness metric
                orig_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                result_gray = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2GRAY)
                
                orig_edges = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
                result_edges = cv2.Laplacian(result_gray, cv2.CV_64F).var()
                sharpness_improvement = ((result_edges - orig_edges) / orig_edges) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Sharpness", f"{orig_edges:.1f}")
                with col2:
                    st.metric("Enhanced Sharpness", f"{result_edges:.1f}")
                with col3:
                    st.metric("Improvement", f"{sharpness_improvement:+.1f}%")
    
    elif method == "Noise Reduction":
        st.subheader("🔇 Noise Reduction")
        
        noise_method = st.selectbox(
            "Noise Reduction Method:",
            ["Bilateral Filter", "Gaussian Blur", "Median Filter", "Non-Local Means"]
        )
        
        if st.button("🧹 Reduce Noise", type="primary"):
            with st.spinner(f"Applying {noise_method}..."):
                method_map = {
                    "Bilateral Filter": "bilateral",
                    "Gaussian Blur": "gaussian", 
                    "Median Filter": "median",
                    "Non-Local Means": "non_local_means"
                }
                result_image = noise_reduction(image, method_map[noise_method])
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Denoised")
                    st.image(result_image, caption=f"{noise_method} Result", use_column_width=True)
                
                # Noise analysis
                st.subheader("📊 Noise Analysis")
                
                # Calculate image quality metrics
                orig_array = np.array(image).astype(np.float32)
                result_array = np.array(result_image).astype(np.float32)
                
                # Standard deviation as noise indicator
                orig_noise = np.std(orig_array)
                result_noise = np.std(result_array)
                noise_reduction_pct = ((orig_noise - result_noise) / orig_noise) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Noise Level", f"{orig_noise:.2f}")
                with col2:
                    st.metric("Reduced Noise Level", f"{result_noise:.2f}")
                with col3:
                    st.metric("Noise Reduction", f"{noise_reduction_pct:.1f}%")
    
    elif method == "White Balance":
        st.subheader("⚖️ White Balance Correction")
        
        wb_method = st.selectbox(
            "White Balance Method:",
            ["Gray World", "White Patch"]
        )
        
        if st.button("🎯 Correct White Balance", type="primary"):
            with st.spinner(f"Applying {wb_method} white balance..."):
                method_map = {
                    "Gray World": "gray_world",
                    "White Patch": "white_patch"
                }
                result_image = white_balance_correction(image, method_map[wb_method])
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("White Balanced")
                    st.image(result_image, caption=f"{wb_method} Correction", use_column_width=True)
                
                # Color temperature analysis
                st.subheader("🌡️ Color Temperature Analysis")
                
                orig_array = np.array(image)
                result_array = np.array(result_image)
                
                # Calculate average color values
                orig_avg = np.mean(orig_array, axis=(0,1))
                result_avg = np.mean(result_array, axis=(0,1))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Color Balance:**")
                    st.write(f"Red: {orig_avg[0]:.1f}")
                    st.write(f"Green: {orig_avg[1]:.1f}")
                    st.write(f"Blue: {orig_avg[2]:.1f}")
                
                with col2:
                    st.write("**Corrected Color Balance:**")
                    st.write(f"Red: {result_avg[0]:.1f}")
                    st.write(f"Green: {result_avg[1]:.1f}")
                    st.write(f"Blue: {result_avg[2]:.1f}")
    
    elif method == "Custom Enhancement":
        st.subheader("🎛️ Custom Enhancement Pipeline")
        
        # Multi-step enhancement
        st.write("**Step 1: Contrast**")
        col1, col2 = st.columns(2)
        with col1:
            apply_clahe = st.checkbox("Apply CLAHE")
            if apply_clahe:
                clahe_clip = st.slider("CLAHE Clip Limit:", 1.0, 10.0, 2.0)
        with col2:
            apply_gamma = st.checkbox("Apply Gamma Correction")
            if apply_gamma:
                gamma_val = st.slider("Gamma Value:", 0.1, 3.0, 1.0)
        
        st.write("**Step 2: Color**")
        col1, col2 = st.columns(2)
        with col1:
            saturation_val = st.slider("Saturation:", 0.0, 2.0, 1.0)
        with col2:
            vibrance_val = st.slider("Vibrance:", -1.0, 1.0, 0.0)
        
        st.write("**Step 3: Sharpening & Noise**")
        col1, col2 = st.columns(2)
        with col1:
            apply_sharpen = st.checkbox("Apply Sharpening")
            if apply_sharpen:
                sharpen_amount = st.slider("Sharpen Amount:", 0.5, 3.0, 1.0)
        with col2:
            apply_denoise = st.checkbox("Apply Denoising")
        
        if st.button("🚀 Apply Custom Enhancement", type="primary"):
            with st.spinner("Applying custom enhancement pipeline..."):
                result_image = image.copy()
                applied_steps = []
                
                # Step 1: Contrast adjustments
                if apply_clahe:
                    result_image = clahe_enhancement(result_image, clahe_clip)
                    applied_steps.append(f"CLAHE (clip: {clahe_clip})")
                
                if apply_gamma:
                    result_image = gamma_correction(result_image, gamma_val)
                    applied_steps.append(f"Gamma Correction ({gamma_val})")
                
                # Step 2: Color adjustments
                if saturation_val != 1.0 or vibrance_val != 0.0:
                    result_image = color_enhancement(result_image, saturation_val, vibrance_val)
                    applied_steps.append(f"Color Enhancement (sat: {saturation_val}, vib: {vibrance_val})")
                
                # Step 3: Sharpening and denoising
                if apply_sharpen:
                    result_image = unsharp_masking(result_image, amount=sharpen_amount)
                    applied_steps.append(f"Unsharp Masking ({sharpen_amount})")
                
                if apply_denoise:
                    result_image = noise_reduction(result_image, "bilateral")
                    applied_steps.append("Bilateral Denoising")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, caption="Input Image", use_column_width=True)
                
                with col2:
                    st.subheader("Enhanced")
                    st.image(result_image, caption="Custom Enhanced", use_column_width=True)
                
                # Enhancement summary
                st.subheader("📋 Enhancement Pipeline Summary")
                if applied_steps:
                    for i, step in enumerate(applied_steps, 1):
                        st.write(f"{i}. {step}")
                else:
                    st.write("No enhancements applied")
                
                # Overall metrics
                st.subheader("📊 Overall Enhancement Metrics")
                
                orig_array = np.array(image)
                result_array = np.array(result_image)
                
                # Calculate various metrics
                orig_contrast = np.std(orig_array)
                result_contrast = np.std(result_array)
                contrast_change = ((result_contrast - orig_contrast) / orig_contrast) * 100
                
                orig_brightness = np.mean(orig_array)
                result_brightness = np.mean(result_array)
                brightness_change = ((result_brightness - orig_brightness) / orig_brightness) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Contrast Change", f"{contrast_change:+.1f}%")
                with col2:
                    st.metric("Brightness Change", f"{brightness_change:+.1f}%")
                with col3:
                    st.metric("Processing Steps", len(applied_steps))
    
    # Download section (common for all methods)
    if 'result_image' in locals():
        st.subheader("💾 Export Enhanced Image")
        import io
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format='PNG')
        
        st.download_button(
            label="📥 Download Enhanced Image",
            data=img_bytes.getvalue(),
            file_name=f"enhanced_{method.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
