"""
Model definition for Oxford 102 Flowers classification.
Uses a pre-trained ResNet model from timm library.
"""
import torch
import torch.nn as nn
import timm


def get_model(num_classes=102, pretrained=True):
    """
    Get a pre-trained ResNet model for flower classification.
    
    Args:
        num_classes (int): Number of flower classes (default: 102 for Oxford Flowers)
        pretrained (bool): Whether to use pretrained weights (default: True)
    
    Returns:
        torch.nn.Module: ResNet model with modified classifier
    """
    # Load pre-trained ResNet50 from timm
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    
    return model
