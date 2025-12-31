# 🌸 Flower Classification Project

A deep learning pipeline to classify 102 different flower species using **ResNet-50** and **Transfer Learning**. Achieved **83% accuracy** on the Oxford 102 Flowers dataset.

## 📋 Project Overview

This project implements a comprehensive flower classification system using state-of-the-art deep learning techniques. The model is trained on the [Oxford 102 Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), which contains images of 102 flower categories commonly occurring in the United Kingdom.

### Key Features

- **Transfer Learning**: Leverages pre-trained ResNet-50 architecture
- **High Accuracy**: Achieves 83% classification accuracy
- **Explainable AI**: Uses Grad-CAM visualization to interpret model predictions
- **Complete Pipeline**: Includes data exploration, training, and evaluation notebooks

## 🔍 Model Insights

Used **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the model's predictions and identify the most important features for each class. Heatmaps confirm the model focuses on unique petal textures and central flower features to distinguish between similar species.

## 📁 Project Structure

```
Flowertest/
├── data/
│   └── flowers-102/          # Oxford 102 Flowers dataset
│       ├── jpg/              # Flower images
│       └── setid.mat         # Train/val/test splits
├── notebooks/
│   ├── 01_EDA.ipynb                           # Exploratory Data Analysis
│   ├── 02_Training.ipynb                      # Model training
│   └── 03_Evaluation_and_Explainability.ipynb # Evaluation & Grad-CAM
├── src/
│   ├── models/
│   │   └── model.py          # ResNet-50 model definition
│   └── utils/
│       └── inference.py      # Inference utilities
├── configs/                  # Configuration files
├── ComputerVision.ipynb     # Main computer vision notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kenooimh/flowerproject.git
   cd flowerproject
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   
   The Oxford 102 Flowers dataset should be placed in the `data/flowers-102/` directory. If not already present, download it from:
   - [Dataset Homepage](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
   - Images: `102flowers.tgz`
   - Labels: `imagelabels.mat`
   - Splits: `setid.mat`

## 📊 Usage

### Training the Model

Open and run the notebooks in order:

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_EDA.ipynb
   ```

2. **Model Training**
   ```bash
   jupyter notebook notebooks/02_Training.ipynb
   ```

3. **Evaluation and Explainability**
   ```bash
   jupyter notebook notebooks/03_Evaluation_and_Explainability.ipynb
   ```

### Running Inference

```python
from src.utils.inference import predict_flower
from PIL import Image

# Load an image
image = Image.open("path/to/flower.jpg")

# Make prediction
prediction = predict_flower(image, model)
print(f"Predicted flower: {prediction}")
```

## 🎯 Results

- **Accuracy**: 83% on test set
- **Model**: ResNet-50 (Transfer Learning)
- **Dataset**: Oxford 102 Flowers (8,189 images across 102 categories)
- **Visualization**: Grad-CAM heatmaps for model interpretability

## 🔧 Model Optimization

Future optimization strategies:
- **Quantization**: FP16 or INT8 for faster inference
- **Pruning**: Remove redundant weights
- **Knowledge Distillation**: Create smaller student models

## 📝 Requirements

Key dependencies:
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Jupyter
- Pillow

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- ResNet-50 architecture by Microsoft Research
- Grad-CAM implementation for explainable AI

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: The trained model file (`best_model.pth`) and large dataset archive (`102flowers.tgz`) are excluded from this repository due to size constraints. Please train the model using the provided notebooks or contact the repository owner for the pre-trained weights.