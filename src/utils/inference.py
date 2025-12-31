"""
Inference utilities for flower classification.
"""
import torch
from torchvision import transforms
from PIL import Image


def predict_flower(image_path, model, device, top_k=3):
    """
    Predict the flower class from an image.
    
    Args:
        image_path (str): Path to the image file
        model (torch.nn.Module): Trained model
        device (torch.device): Device to run inference on
        top_k (int): Number of top predictions to return
    
    Returns:
        dict: Dictionary containing predicted_class, confidence_score, and top_k predictions
    """
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to lists
        top_probs = top_probs.cpu().numpy()[0].tolist()
        top_indices = top_indices.cpu().numpy()[0].tolist()
        
        # Create top_k list with class and confidence
        top_k_predictions = [
            {"class": idx, "confidence": prob}
            for idx, prob in zip(top_indices, top_probs)
        ]
    
    return {
        "predicted_class": top_indices[0],
        "confidence_score": top_probs[0],
        "top_3": top_k_predictions
    }
