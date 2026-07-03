# 🫁 Pneumonia Detection AI

A deep learning-based application for detecting pneumonia from chest X-ray images using DINOv2 vision transformer and custom CNN heads. This project includes both a trained model and a user-friendly Streamlit web interface for inference and demonstration.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## 🎯 Project Overview

This project implements an AI-powered pneumonia detection system that:
- **Classifies** chest X-ray images as Normal or Pneumonia
- **Uses** DINOv2 (self-supervised vision transformer) as backbone
- **Combines** transformer features with CNN for robust classification
- **Provides** an interactive web interface for easy access
- **Offers** model interpretability through confusion matrices and performance metrics

**Target Users:** Medical professionals, students, and researchers

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│     Streamlit Web Interface (app.py)    │
│  - Upload X-ray images                  │
│  - Real-time predictions                │
│  - Performance visualizations           │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼───────────┐
        │ Model Loading    │
        │ (model_best.pth) │
        └──────┬───────────┘
               │
    ┌──────────▼──────────────┐
    │  DINOv2 + Custom CNN    │
    │  - Vision Transformer   │
    │  - Feature Extraction   │
    │  - Classification Head  │
    └────────────────────────┘
```

### Model Components
1. **Backbone:** DINOv2-small (facebook/dinov2-small) - Pre-trained vision transformer
2. **Feature Enhancement:** Custom CNN module with depthwise separable convolutions
3. **Classification:** Binary classifier (Normal vs Pneumonia)

---


### Features
- 📤 **Upload X-ray Images** - PNG, JPG, JPEG formats
- 🤖 **Real-time Predictions** - Get results in seconds
- 📊 **Model Predictions** - View confidence scores
- 📈 **Performance Metrics** - See model accuracy and confusion matrix
- 🎓 **Explainability** - Understand model decisions

---


## 📊 Results

The trained model achieves:
- **Accuracy:** [Check `outputs/results.json`]
- **Precision:** [Check `outputs/results.json`]
- **Recall:** [Check `outputs/results.json`]
- **F1-Score:** [Check `outputs/results.json`]

See visualizations in `outputs/`:
- `loss.png` - Training loss curves
- `accuracy.png` - Accuracy trends
- `confusion_matrix.png` - Model confusion matrix

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch |
| **Vision Backbone** | DINOv2 (Facebook Research) |
| **Preprocessing** | Torchvision, Pillow, OpenCV |
| **Web Framework** | Streamlit |
| **Metrics** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## 📦 Dependencies

**Core Requirements:**
- `numpy>=1.21.0` - Numerical computing
- `torch>=1.11.0` - Deep learning framework
- `torchvision>=0.12.0` - Computer vision utilities
- `transformers>=4.30.0` - Pre-trained models (DINOv2)
- `Pillow>=9.0.0` - Image processing
- `scikit-learn>=1.0.0` - ML metrics
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `pandas` - Data handling
- `streamlit` - Web interface

---


## 📝 License

This project is created for educational purposes.

---

## 👨‍💻 Author

Created as an AI Project for my AIP project.

---

## 🤝 Support

For questions or issues:
1. Check the model path: `outputs/model_best.pth`
2. Verify all dependencies: `pip install -r requirements.txt`
3. Ensure GPU drivers are installed (optional but recommended)

---

