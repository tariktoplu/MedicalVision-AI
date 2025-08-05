# 🏥 MedicalVision-AI

> **Advanced Medical Imaging Analysis System** - Professional desktop application for automated brain CT/MR image analysis using deep learning ensemble models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)

## 🎯 Overview

MedicalVision-AI is a state-of-the-art desktop application designed for medical professionals to analyze brain imaging data. The system uses a 5-fold ensemble CNN-LSTM model with SE (Squeeze-and-Excitation) attention mechanism to classify stroke lesion phases.

### 🔬 Medical Classifications
- **HiperakutAkut** (Hyperacute/Acute)
- **Subakut** (Subacute) 
- **NormalKronik** (Normal/Chronic)

## ✨ Features

### 🖥️ **Modern Desktop Interface**
- Professional PyQt5 GUI with responsive design
- Intuitive workflow: Modality → Analysis Mode → Results
- Real-time image preview and processing status

### 🧠 **Advanced AI Model**
- **5-Fold Ensemble Learning** for higher accuracy
- **CNN-LSTM Architecture** with SE Attention blocks
- **GPU/CPU Auto-detection** for optimal performance
- **Focal Loss** optimization for imbalanced medical data

### 📁 **Multi-Format Support**
- **DICOM** (.dcm) - Native medical imaging format
- **Standard Images** (.png, .jpg, .jpeg, .bmp)
- **Batch Processing** - Analyze multiple files simultaneously
- **Automatic Preprocessing** - Normalization and resizing

### 🎛️ **Analysis Modes**
- **Single Analysis**: Individual file processing with detailed results
- **Multi Analysis**: Batch processing with tabular results
- **Progress Tracking**: Real-time processing status
- **Error Handling**: Robust error management per file

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/MedicalVision-AI.git
cd MedicalVision-AI
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv medical_ai_env
medical_ai_env\Scripts\activate

# macOS/Linux
python3 -m venv medical_ai_env
source medical_ai_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Model Files
Create the model directory and place your trained models:
```bash
mkdir Models
```

Place your 5-fold model files in the `Models/` directory:
```
Models/
├── best_model_fold_0.pt
├── best_model_fold_1.pt
├── best_model_fold_2.pt
├── best_model_fold_3.pt
└── best_model_fold_4.pt
```

## 🎮 Usage

### Launch Application
```bash
python medical_analyzer.py
```

### Step-by-Step Workflow

1. **Select Modality**
   - Choose between BT (CT) or MR imaging

2. **Choose Analysis Mode**
   - **Single Analysis**: For individual file processing
   - **Multi Analysis**: For batch processing multiple files

3. **Upload Files**
   - Support for DICOM (.dcm) and standard image formats
   - Drag & drop functionality (in single mode)

4. **View Results**
   - **Prediction**: Primary classification result
   - **Confidence**: Prediction confidence percentage
   - **All Probabilities**: Complete class probability breakdown

### Single Analysis Features
- 🖼️ **Image Preview**: Real-time visualization
- 📊 **Detailed Results**: Comprehensive analysis output  
- ⚡ **Instant Processing**: Fast single-file analysis

### Multi Analysis Features
- 📁 **Batch Upload**: Process multiple files at once
- 📋 **Table View**: Organized results in tabular format
- 📈 **Progress Tracking**: Real-time processing status
- 🔄 **Parallel Processing**: Efficient multi-file handling

## 🔧 Technical Architecture

### Model Architecture
```
Input (256x256x16) → 3D CNN → SE Attention → MaxPool → 
3D CNN → SE Attention → MaxPool → LSTM → Dense → Output (3 classes)
```

### Key Components
- **SE Blocks**: Squeeze-and-Excitation attention mechanism
- **3D Convolutions**: Spatial-temporal feature extraction
- **LSTM**: Sequential pattern recognition
- **Focal Loss**: Handles class imbalance in medical data

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB+ for models and dependencies
- **GPU**: CUDA-compatible (optional, improves speed 10x+)

## 📊 Performance Metrics

The ensemble model achieves high accuracy through:
- ✅ **5-Fold Cross-Validation** training
- ✅ **Ensemble Averaging** for robust predictions
- ✅ **Class-Weighted Sampling** for balanced learning
- ✅ **Early Stopping** to prevent overfitting

## 🛠️ Development

### Project Structure
```
MedicalVision-AI/
├── medical_analyzer.py      # Main application
├── train.py                 # Model training script
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── Models/          # Model files directory
│   ├── best_model_fold_0.pt
│   └── ...
└── assets/                 # Application assets (optional)
```

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 🔍 Troubleshooting

### Common Issues

**Model Loading Error**
```bash
Error: Hiçbir model dosyası bulunamadı!
```
**Solution**: Ensure model files are in `Models/` directory with correct naming.

**CUDA Out of Memory**
```bash
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU mode (automatic fallback).

**PyQt5 Import Error**
```bash
ModuleNotFoundError: No module named 'PyQt5'
```
**Solution**: Install PyQt5 with `pip install PyQt5` or use conda environment.

### System Requirements Check
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- DICOM community for medical imaging standards
- PyQt5 developers for the GUI framework
- Medical imaging research community

## 📧 Contact

- **Author**: Tarık Toplu
- **Email**: tarikttoplu@gmail.com
- **Project Link**: [https://github.com/yourusername/MedicalVision-AI](https://github.com/tariktoplu/MedicalVision-AI)

---

⭐ **Star this repository if it helped you!**

🔬 **Built for medical professionals, by AI researchers**
