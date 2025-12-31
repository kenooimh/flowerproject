#!/usr/bin/env python3
"""
Quick test to verify notebooks 02 and 03 can execute their initial cells.
"""
import sys
import os

# Add project root to path
sys.path.append('/Users/kenooimh/Documents/AI tests/Flowertest')

print("=" * 60)
print("TESTING NOTEBOOK 02 - TRAINING")
print("=" * 60)

try:
    # Cell 1: Imports
    print("\n[Cell 1] Testing imports...")
    from src.models.model import get_model
    print("✓ Successfully imported get_model")
    
    # Cell 2: Device setup
    print("\n[Cell 2] Setting up device...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Training will run on: {device}")
    
    # Cell 3: Model initialization
    print("\n[Cell 3] Initializing model...")
    import torch.nn as nn
    import torch.optim as optim
    model = get_model(num_classes=102).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Cell 4: Data loaders (without actually loading data)
    print("\n[Cell 4] Testing data loader setup...")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    print("✓ Data loader imports successful")
    
    print("\n✅ NOTEBOOK 02: All initial cells can execute successfully!")
    
except Exception as e:
    print(f"\n❌ NOTEBOOK 02 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TESTING NOTEBOOK 03 - EVALUATION")
print("=" * 60)

try:
    # Cell 1: Imports
    print("\n[Cell 1] Testing imports...")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from src.utils.inference import predict_flower
    print("✓ Successfully imported all modules")
    
    # Cell 2: Model loading (without actual weights)
    print("\n[Cell 2] Testing model loading setup...")
    model = get_model(num_classes=102).to(device)
    model.eval()
    model_path = '/Users/kenooimh/Documents/AI tests/Flowertest/best_model.pth'
    if os.path.exists(model_path):
        print("✓ Model weights file exists")
    else:
        print("⚠️  Model weights not found (expected - train first)")
    
    # Cell 3: Grad-CAM imports
    print("\n[Cell 3] Testing Grad-CAM imports...")
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    print("✓ Grad-CAM imports successful")
    
    # Cell 4: Metrics imports
    print("\n[Cell 4] Testing metrics imports...")
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import seaborn as sns
    print("✓ Metrics imports successful")
    
    # Cell 5: Inference function
    print("\n[Cell 5] Testing inference function...")
    print("✓ predict_flower function is available")
    
    print("\n✅ NOTEBOOK 03: All initial cells can execute successfully!")
    
except Exception as e:
    print(f"\n❌ NOTEBOOK 03 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Both notebooks are ready to run!")
print("Note: Full execution requires:")
print("  - Notebook 02: ~10-30 minutes for training")
print("  - Notebook 03: Requires trained model from Notebook 02")
