"""
Image utility functions for loading, processing, and saving images.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import io
import base64
from typing import Tuple, Optional, Union

def load_image(uploaded_file) -> Image.Image:
    """
    Load image from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        PIL Image object
    """
    try:
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def save_image(image: Image.Image, filename: str, format: str = 'PNG') -> bytes:
    """
    Save PIL Image to bytes.
    
    Args:
        image: PIL Image object
        filename: Output filename
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Image bytes
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=format)
    return img_bytes.getvalue()

def resize_image(image: Image.Image, max_size: Tuple[int, int] = (800, 600), 
                maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        image: PIL Image object
        
    Returns:
        OpenCV image array (BGR format)
    """
    # Convert PIL to RGB array
    rgb_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array

def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL format.
    
    Args:
        cv_image: OpenCV image array
        
    Returns:
        PIL Image object
    """
    # Handle different input formats
    if len(cv_image.shape) == 3:
        # Color image - convert BGR to RGB
        rgb_array = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale image
        rgb_array = cv_image
    
    # Convert to PIL
    return Image.fromarray(rgb_array)

def normalize_image(image: Union[Image.Image, np.ndarray], 
                   target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Normalize image for model input.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target (width, height) for resizing
        
    Returns:
        Normalized numpy array
    """
    # Convert PIL to numpy if necessary
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Resize if target size specified
    if target_size:
        if isinstance(image, Image.Image):
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(image)
        else:
            img_array = cv2.resize(img_array, target_size)
    
    # Normalize to 0-1 range
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array

def create_image_grid(images: list, grid_size: Tuple[int, int], 
                     cell_size: Tuple[int, int] = (200, 200)) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        grid_size: (rows, cols) for the grid
        cell_size: Size of each cell in the grid
        
    Returns:
        Combined PIL Image
    """
    rows, cols = grid_size
    cell_width, cell_height = cell_size
    
    # Create blank canvas
    grid_width = cols * cell_width
    grid_height = rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Place images in grid
    for idx, img in enumerate(images[:rows * cols]):
        row = idx // cols
        col = idx % cols
        
        # Resize image to fit cell
        img_resized = img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        
        # Calculate position
        x = col * cell_width
        y = row * cell_height
        
        # Paste image
        grid_image.paste(img_resized, (x, y))
    
    return grid_image

def display_images_side_by_side(images: list, captions: list = None, 
                               width: int = 300) -> None:
    """
    Display multiple images side by side in Streamlit.
    
    Args:
        images: List of PIL Images or numpy arrays
        captions: List of captions for each image
        width: Width for each image
    """
    if not images:
        return
    
    # Create columns
    cols = st.columns(len(images))
    
    # Display each image
    for i, (col, img) in enumerate(zip(cols, images)):
        with col:
            caption = captions[i] if captions and i < len(captions) else f"Image {i+1}"
            
            # Convert numpy array to PIL if necessary
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Color image
                    img = Image.fromarray(img.astype(np.uint8))
                elif len(img.shape) == 2:
                    # Grayscale image
                    img = Image.fromarray(img.astype(np.uint8), mode='L')
                else:
                    st.error(f"Unsupported image shape: {img.shape}")
                    continue
            
            st.image(img, caption=caption, width=width)

def apply_image_augmentation(image: Image.Image, augmentation_type: str) -> Image.Image:
    """
    Apply image augmentation transformations.
    
    Args:
        image: PIL Image object
        augmentation_type: Type of augmentation to apply
        
    Returns:
        Augmented PIL Image
    """
    if augmentation_type == "horizontal_flip":
        return ImageOps.mirror(image)
    elif augmentation_type == "vertical_flip":
        return ImageOps.flip(image)
    elif augmentation_type == "rotate_90":
        return image.rotate(90, expand=True)
    elif augmentation_type == "rotate_180":
        return image.rotate(180, expand=True)
    elif augmentation_type == "rotate_270":
        return image.rotate(270, expand=True)
    elif augmentation_type == "grayscale":
        return ImageOps.grayscale(image).convert('RGB')
    elif augmentation_type == "invert":
        return ImageOps.invert(image)
    elif augmentation_type == "posterize":
        return ImageOps.posterize(image, 4)
    elif augmentation_type == "solarize":
        return ImageOps.solarize(image, 128)
    else:
        return image

def calculate_image_statistics(image: Image.Image) -> dict:
    """
    Calculate various statistics for an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing image statistics
    """
    img_array = np.array(image)
    
    stats = {
        'width': image.width,
        'height': image.height,
        'channels': len(img_array.shape),
        'total_pixels': img_array.size,
        'mean_brightness': np.mean(img_array),
        'std_brightness': np.std(img_array),
        'min_value': np.min(img_array),
        'max_value': np.max(img_array),
        'file_size_estimate': len(save_image(image, 'temp.png'))
    }
    
    # Add channel-specific statistics for color images
    if len(img_array.shape) == 3:
        for i, channel in enumerate(['red', 'green', 'blue']):
            stats[f'{channel}_mean'] = np.mean(img_array[:, :, i])
            stats[f'{channel}_std'] = np.std(img_array[:, :, i])
    
    return stats

def create_image_comparison(original: Image.Image, processed: Image.Image, 
                          title: str = "Image Comparison") -> None:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        original: Original PIL Image
        processed: Processed PIL Image
        title: Title for the comparison
    """
    st.subheader(title)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original, caption="Original", use_container_width=True)
        
        # Original image stats
        orig_stats = calculate_image_statistics(original)
        st.write(f"**Size:** {orig_stats['width']} × {orig_stats['height']}")
        st.write(f"**Mean Brightness:** {orig_stats['mean_brightness']:.1f}")
    
    with col2:
        st.image(processed, caption="Processed", use_container_width=True)
        
        # Processed image stats
        proc_stats = calculate_image_statistics(processed)
        st.write(f"**Size:** {proc_stats['width']} × {proc_stats['height']}")
        st.write(f"**Mean Brightness:** {proc_stats['mean_brightness']:.1f}")
        
        # Calculate difference
        brightness_diff = proc_stats['mean_brightness'] - orig_stats['mean_brightness']
        st.write(f"**Brightness Change:** {brightness_diff:+.1f}")

def extract_image_patches(image: Image.Image, patch_size: Tuple[int, int], 
                         stride: int = None) -> list:
    """
    Extract patches from an image for analysis.
    
    Args:
        image: PIL Image object
        patch_size: (width, height) of each patch
        stride: Step size between patches (defaults to patch width)
        
    Returns:
        List of PIL Image patches
    """
    if stride is None:
        stride = patch_size[0]
    
    img_array = np.array(image)
    patches = []
    
    patch_w, patch_h = patch_size
    
    for y in range(0, img_array.shape[0] - patch_h + 1, stride):
        for x in range(0, img_array.shape[1] - patch_w + 1, stride):
            patch = img_array[y:y+patch_h, x:x+patch_w]
            patches.append(Image.fromarray(patch))
    
    return patches

def blend_images(image1: Image.Image, image2: Image.Image, 
                alpha: float = 0.5) -> Image.Image:
    """
    Blend two images together.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        alpha: Blending factor (0.0 = only image1, 1.0 = only image2)
        
    Returns:
        Blended PIL Image
    """
    # Ensure both images are the same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
    
    # Blend images
    blended = Image.blend(image1, image2, alpha)
    
    return blended

def convert_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    img_bytes = save_image(image, "temp.png")
    return base64.b64encode(img_bytes).decode()

def validate_image(image: Image.Image, max_size: int = 10485760) -> bool:
    """
    Validate if image meets requirements.
    
    Args:
        image: PIL Image object
        max_size: Maximum file size in bytes
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if image is valid
        if image is None:
            return False
        
        # Check dimensions
        if image.width <= 0 or image.height <= 0:
            return False
        
        # Check file size (approximate)
        img_bytes = save_image(image, "temp.png")
        if len(img_bytes) > max_size:
            return False
        
        return True
    except Exception:
        return False
