"""
Model configurations and settings for the Computer Vision application.
"""

# Model configurations
MODEL_CONFIGS = {
    'ResNet50': {
        'name': 'ResNet50',
        'description': 'Deep residual network with 50 layers, excellent for image classification',
        'input_size': (224, 224),
        'preprocessing': 'imagenet',
        'output_classes': 1000,
        'architecture': 'Residual Network',
        'parameters': '25.6M',
        'top1_accuracy': 76.1,
        'top5_accuracy': 92.9,
        'use_cases': ['Image Classification', 'Feature Extraction', 'Transfer Learning'],
        'strengths': ['Very deep architecture', 'Skip connections prevent vanishing gradients', 'Good accuracy'],
        'weaknesses': ['Large model size', 'Computationally intensive']
    },
    'VGG16': {
        'name': 'VGG16',
        'description': 'Visual Geometry Group network with 16 layers, simple but effective architecture',
        'input_size': (224, 224),
        'preprocessing': 'imagenet',
        'output_classes': 1000,
        'architecture': 'Convolutional Neural Network',
        'parameters': '138M',
        'top1_accuracy': 71.3,
        'top5_accuracy': 90.1,
        'use_cases': ['Image Classification', 'Feature Extraction', 'Style Transfer'],
        'strengths': ['Simple architecture', 'Good feature extraction', 'Well-established'],
        'weaknesses': ['Very large model size', 'Slower inference', 'Many parameters']
    },
    'MobileNetV2': {
        'name': 'MobileNetV2',
        'description': 'Efficient mobile-optimized network with depthwise separable convolutions',
        'input_size': (224, 224),
        'preprocessing': 'imagenet',
        'output_classes': 1000,
        'architecture': 'Inverted Residual Network',
        'parameters': '3.5M',
        'top1_accuracy': 71.8,
        'top5_accuracy': 90.6,
        'use_cases': ['Mobile Applications', 'Edge Computing', 'Real-time Classification'],
        'strengths': ['Very efficient', 'Small model size', 'Fast inference'],
        'weaknesses': ['Lower accuracy than larger models', 'Limited feature capacity']
    },
    'InceptionV3': {
        'name': 'InceptionV3',
        'description': 'Inception architecture with factorized convolutions and auxiliary classifiers',
        'input_size': (299, 299),
        'preprocessing': 'imagenet',
        'output_classes': 1000,
        'architecture': 'Inception Network',
        'parameters': '23.9M',
        'top1_accuracy': 77.9,
        'top5_accuracy': 93.7,
        'use_cases': ['Image Classification', 'Computer Vision Research', 'High Accuracy Tasks'],
        'strengths': ['High accuracy', 'Efficient architecture', 'Multi-scale features'],
        'weaknesses': ['Complex architecture', 'Different input size', 'Memory intensive']
    },
    'DenseNet121': {
        'name': 'DenseNet121',
        'description': 'Densely connected network with feature reuse and efficient parameter usage',
        'input_size': (224, 224),
        'preprocessing': 'imagenet',
        'output_classes': 1000,
        'architecture': 'Dense Network',
        'parameters': '8.0M',
        'top1_accuracy': 74.4,
        'top5_accuracy': 92.2,
        'use_cases': ['Image Classification', 'Feature Learning', 'Medical Imaging'],
        'strengths': ['Parameter efficient', 'Strong feature reuse', 'Good gradient flow'],
        'weaknesses': ['Memory intensive during training', 'Complex connectivity']
    },
    'EfficientNetB0': {
        'name': 'EfficientNetB0',
        'description': 'Efficiently scaled network balancing depth, width, and resolution',
        'input_size': (224, 224),
        'preprocessing': 'imagenet',
        'output_classes': 1000,
        'architecture': 'EfficientNet',
        'parameters': '5.3M',
        'top1_accuracy': 77.1,
        'top5_accuracy': 93.3,
        'use_cases': ['Efficient Classification', 'Resource-Constrained Environments', 'AutoML'],
        'strengths': ['Excellent accuracy/efficiency trade-off', 'Systematic scaling', 'SOTA efficiency'],
        'weaknesses': ['Complex training procedure', 'Requires careful hyperparameter tuning']
    }
}

# Object Detection Models
OBJECT_DETECTION_CONFIGS = {
    'SSD_MobileNet': {
        'name': 'SSD MobileNet',
        'description': 'Single Shot Detector with MobileNet backbone for efficient object detection',
        'input_size': (300, 300),
        'output_classes': 80,  # COCO dataset classes
        'architecture': 'Single Shot Detector',
        'use_cases': ['Real-time Detection', 'Mobile Applications', 'Edge Computing'],
        'strengths': ['Fast inference', 'Single forward pass', 'Good for real-time'],
        'weaknesses': ['Lower accuracy on small objects', 'Limited to grid-based detection']
    },
    'YOLOv3': {
        'name': 'YOLO v3',
        'description': 'You Only Look Once version 3 for real-time object detection',
        'input_size': (416, 416),
        'output_classes': 80,
        'architecture': 'YOLO',
        'use_cases': ['Real-time Detection', 'Video Analysis', 'Surveillance'],
        'strengths': ['Very fast', 'End-to-end training', 'Good for real-time'],
        'weaknesses': ['Struggles with small objects', 'Localization errors']
    },
    'Faster_RCNN': {
        'name': 'Faster R-CNN',
        'description': 'Two-stage detector with region proposal network for high accuracy',
        'input_size': (800, 600),
        'output_classes': 80,
        'architecture': 'Two-Stage Detector',
        'use_cases': ['High Accuracy Detection', 'Research', 'Quality Applications'],
        'strengths': ['High accuracy', 'Good localization', 'Handles various object sizes'],
        'weaknesses': ['Slower inference', 'Complex architecture', 'More memory intensive']
    }
}

# Segmentation Models
SEGMENTATION_CONFIGS = {
    'DeepLabV3': {
        'name': 'DeepLab v3',
        'description': 'Semantic segmentation with atrous convolution and spatial pyramid pooling',
        'input_size': (513, 513),
        'output_classes': 21,  # PASCAL VOC classes
        'architecture': 'Encoder-Decoder',
        'use_cases': ['Semantic Segmentation', 'Scene Understanding', 'Medical Imaging'],
        'strengths': ['Good boundary detection', 'Multi-scale features', 'State-of-the-art accuracy'],
        'weaknesses': ['Computationally intensive', 'Large memory requirements']
    },
    'UNet': {
        'name': 'U-Net',
        'description': 'U-shaped network for biomedical image segmentation with skip connections',
        'input_size': (256, 256),
        'output_classes': 'variable',
        'architecture': 'U-Net',
        'use_cases': ['Medical Imaging', 'Biomedical Segmentation', 'Small Dataset Tasks'],
        'strengths': ['Works with small datasets', 'Good for medical images', 'Precise segmentation'],
        'weaknesses': ['Limited to specific domains', 'Requires careful data preparation']
    },
    'MaskRCNN': {
        'name': 'Mask R-CNN',
        'description': 'Instance segmentation extending Faster R-CNN with mask prediction',
        'input_size': (800, 600),
        'output_classes': 80,
        'architecture': 'Instance Segmentation',
        'use_cases': ['Instance Segmentation', 'Object Detection + Segmentation', 'Research'],
        'strengths': ['Instance-level segmentation', 'High quality masks', 'Versatile'],
        'weaknesses': ['Very computationally intensive', 'Complex training', 'Large model']
    }
}

# Style Transfer Models
STYLE_TRANSFER_CONFIGS = {
    'Neural_Style_Transfer': {
        'name': 'Neural Style Transfer',
        'description': 'Original neural style transfer using VGG features and optimization',
        'input_size': (512, 512),
        'architecture': 'Optimization-based',
        'use_cases': ['Artistic Style Transfer', 'Creative Applications', 'Research'],
        'strengths': ['High quality results', 'Flexible style control', 'Well-established'],
        'weaknesses': ['Very slow', 'Requires optimization per image', 'Memory intensive']
    },
    'Fast_Style_Transfer': {
        'name': 'Fast Style Transfer',
        'description': 'Feed-forward network for real-time style transfer',
        'input_size': (256, 256),
        'architecture': 'Feed-forward Network',
        'use_cases': ['Real-time Style Transfer', 'Video Stylization', 'Interactive Applications'],
        'strengths': ['Fast inference', 'Single forward pass', 'Good for real-time'],
        'weaknesses': ['One style per model', 'Lower quality than optimization-based', 'Style-specific training']
    },
    'AdaIN': {
        'name': 'AdaIN',
        'description': 'Adaptive Instance Normalization for arbitrary style transfer',
        'input_size': (512, 512),
        'architecture': 'Encoder-Decoder with AdaIN',
        'use_cases': ['Arbitrary Style Transfer', 'Style Interpolation', 'Creative Tools'],
        'strengths': ['Single model for multiple styles', 'Style interpolation', 'Good quality'],
        'weaknesses': ['Moderate speed', 'Style control limitations', 'Training complexity']
    }
}

# Default settings
DEFAULT_MODEL_SETTINGS = {
    'classification': {
        'confidence_threshold': 0.1,
        'top_k_predictions': 5,
        'batch_size': 1,
        'preprocessing': 'imagenet'
    },
    'object_detection': {
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4,
        'max_detections': 100,
        'input_size': (416, 416)
    },
    'segmentation': {
        'confidence_threshold': 0.5,
        'output_stride': 16,
        'apply_crf': False,
        'input_size': (513, 513)
    },
    'style_transfer': {
        'style_weight': 1.0,
        'content_weight': 1.0,
        'style_layers': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        'content_layers': ['conv4_2'],
        'num_iterations': 1000
    }
}

# Hardware requirements
HARDWARE_REQUIREMENTS = {
    'minimum': {
        'ram_gb': 4,
        'gpu_memory_gb': 2,
        'cpu_cores': 2,
        'storage_gb': 10
    },
    'recommended': {
        'ram_gb': 8,
        'gpu_memory_gb': 6,
        'cpu_cores': 4,
        'storage_gb': 20
    },
    'optimal': {
        'ram_gb': 16,
        'gpu_memory_gb': 11,
        'cpu_cores': 8,
        'storage_gb': 50
    }
}

# Performance benchmarks (approximate inference times in ms)
PERFORMANCE_BENCHMARKS = {
    'ResNet50': {'cpu': 45, 'gpu': 5},
    'VGG16': {'cpu': 85, 'gpu': 8},
    'MobileNetV2': {'cpu': 15, 'gpu': 2},
    'InceptionV3': {'cpu': 55, 'gpu': 6},
    'DenseNet121': {'cpu': 40, 'gpu': 4},
    'EfficientNetB0': {'cpu': 25, 'gpu': 3}
}

# Model categories for UI organization
MODEL_CATEGORIES = {
    'Lightweight': ['MobileNetV2', 'EfficientNetB0'],
    'High Accuracy': ['InceptionV3', 'ResNet50'],
    'Classic': ['VGG16'],
    'Research': ['DenseNet121'],
    'Mobile': ['MobileNetV2'],
    'Production': ['ResNet50', 'EfficientNetB0']
}

def get_model_config(model_name: str, category: str = 'classification') -> dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        category: Category of the model (classification, object_detection, etc.)
        
    Returns:
        Model configuration dictionary
    """
    config_map = {
        'classification': MODEL_CONFIGS,
        'object_detection': OBJECT_DETECTION_CONFIGS,
        'segmentation': SEGMENTATION_CONFIGS,
        'style_transfer': STYLE_TRANSFER_CONFIGS
    }
    
    configs = config_map.get(category, MODEL_CONFIGS)
    return configs.get(model_name, {})

def get_recommended_models(use_case: str) -> list:
    """
    Get recommended models for a specific use case.
    
    Args:
        use_case: Use case description
        
    Returns:
        List of recommended model names
    """
    recommendations = {
        'mobile': ['MobileNetV2', 'EfficientNetB0'],
        'high_accuracy': ['InceptionV3', 'ResNet50'],
        'real_time': ['MobileNetV2', 'YOLOv3'],
        'research': ['DenseNet121', 'InceptionV3'],
        'production': ['ResNet50', 'EfficientNetB0'],
        'edge_computing': ['MobileNetV2'],
        'cloud_deployment': ['ResNet50', 'InceptionV3']
    }
    
    return recommendations.get(use_case.lower(), ['ResNet50', 'MobileNetV2'])

def get_hardware_recommendation(model_name: str) -> dict:
    """
    Get hardware recommendation for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Hardware recommendation dictionary
    """
    lightweight_models = ['MobileNetV2', 'EfficientNetB0']
    heavy_models = ['VGG16', 'InceptionV3']
    
    if model_name in lightweight_models:
        return HARDWARE_REQUIREMENTS['minimum']
    elif model_name in heavy_models:
        return HARDWARE_REQUIREMENTS['optimal']
    else:
        return HARDWARE_REQUIREMENTS['recommended']
