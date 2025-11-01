# 🎯 FLIR ADAS Target Recognition System

Multi-Modal Object Detection with Generative Inpainting for Automotive Applications

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

This project implements a state-of-the-art target recognition system optimized for FLIR ADAS (Advanced Driver Assistance Systems) dataset. It combines:

- **YOLOv8 Object Detection**: Fast and accurate real-time detection
- **Multi-Modal Processing**: Supports both RGB and Thermal imagery
- **Generative Inpainting**: AI-powered reconstruction of occluded/degraded targets
- **Interactive Web Interface**: User-friendly Streamlit application

### Key Highlights

✅ **15 Object Classes**: Person, Bike, Car, Motor, Bus, Train, Truck, Light, Hydrant, Sign, Dog, Skateboard, Stroller, Scooter, Other Vehicle

✅ **Real-time Performance**: GPU-accelerated inference (~30+ FPS on RTX 3050)

✅ **Production Ready**: Trained model with 47% mAP@0.5, 93% recall

---

## 🎨 Features

### Detection System
- **YOLOv8-based** detection optimized for automotive scenarios
- **Multi-modal support**: RGB, Thermal, and Fused inputs
- **Real-time processing** with GPU acceleration
- **Adjustable confidence** thresholds
- **Object tracking** across video frames

### Inpainting Module
- **Stable Diffusion** powered reconstruction
- **Automatic occlusion detection**
- **FLIR-specific prompts** for better results
- **Smart masking** based on image analysis

### Web Interface
- **Interactive Streamlit app**
- **Single image and batch processing**
- **Real-time visualization**
- **Downloadable results**
- **Performance metrics display**

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **CPU**: Intel i5 or equivalent
- **RAM**: 8GB
- **Storage**: 20GB free space
- **Python**: 3.11+

### Recommended Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3050 or better)
- **CUDA**: 11.8 or 12.1+
- **RAM**: 16GB
- **Storage**: 50GB SSD

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/flir-adas-target-recognition.git
cd flir-adas-target-recognition
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv target_recognition_env
.\target_recognition_env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv target_recognition_env
source target_recognition_env/bin/activate
```

### Step 3: Install Dependencies

**With GPU (CUDA 12.1):**
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install ultralytics opencv-python numpy pyyaml tqdm streamlit pillow
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pyyaml tqdm streamlit pillow
```

**Optional (for Inpainting):**
```bash
pip install diffusers transformers accelerate
```

### Step 4: Verify Installation

```bash
python test_pipeline.py
```

---

## 🎬 Quick Start

### 1. Prepare Dataset

Download the FLIR ADAS dataset and process it:

```bash
python setup_flir_dataset.py \
    --flir-dir /path/to/FLIR_ADAS_v2 \
    --output ./processed_data \
    --fusion rgb_only
```

### 2. Train Model

```bash
python train_pipeline.py
```

Training will take ~2-3 hours on GPU (30 minutes per 10 epochs).

### 3. Test Model

```bash
python quick_visual_test.py
```

### 4. Launch Web App

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📖 Usage

### Command Line Interface

#### Single Image Detection

```bash
python inference_demo.py \
    --model runs/detect/multimodal_yolo/weights/best.pt \
    --mode folder \
    --input test_images/ \
    --output results/
```

#### Webcam Demo

```bash
python inference_demo.py \
    --model runs/detect/multimodal_yolo/weights/best.pt \
    --mode webcam
```

#### Video Processing

```bash
python complete_pipeline.py \
    --detector runs/detect/multimodal_yolo/weights/best.pt \
    --rgb-input video.mp4 \
    --output output.mp4
```

#### With Inpainting

```bash
python complete_pipeline.py \
    --detector runs/detect/multimodal_yolo/weights/best.pt \
    --rgb-input image.jpg \
    --use-inpainting \
    --output result.jpg
```

### Web Interface

1. **Launch App**: `streamlit run app.py`
2. **Select Model**: Choose from available trained models
3. **Configure Settings**: Adjust confidence threshold and inpainting options
4. **Upload Images**: Single or batch processing
5. **Download Results**: Get annotated images

---

## 📁 Project Structure

```
flir-adas-target-recognition/
│
├── 📄 app.py                      # Streamlit web application
├── 📄 detection_model.py          # YOLOv8 detection module
├── 📄 inpainting_module.py        # Stable Diffusion inpainting
├── 📄 complete_pipeline.py        # Full processing pipeline
├── 📄 train_pipeline.py           # Training orchestration
├── 📄 inference_demo.py           # CLI demo interface
├── 📄 setup_flir_dataset.py       # Dataset preprocessing
├── 📄 test_pipeline.py            # System validation
├── 📄 quick_visual_test.py        # Quick model testing
├── 📄 config.yaml                 # Training configuration
├── 📄 README.md                   # This file
│
├── 📁 processed_data/             # Processed FLIR dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
│
├── 📁 runs/                       # Training outputs
│   └── detect/
│       └── multimodal_yolo/
│           └── weights/
│               └── best.pt        # Trained model
│
├── 📁 results/                    # Saved results
├── 📁 models/                     # Model checkpoints
├── 📁 logs/                       # Training logs
└── 📁 target_recognition_env/     # Virtual environment
```

---

## 📊 Model Performance

### Trained Model Stats (23 Epochs)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 47% |
| **mAP@0.5:0.95** | 29% |
| **Precision** | 71% |
| **Recall** | 93% |
| **Inference Speed** | ~35ms (RTX 3050) |
| **FPS** | ~28 FPS |

### Dataset Statistics

- **Training Images**: 10,319
- **Validation Images**: 1,085
- **Classes**: 15 (FLIR ADAS standard)
- **Image Size**: 640x640

### Class Distribution

| Class | Count | Class | Count |
|-------|-------|-------|-------|
| Person | 3,245 | Bike | 892 |
| Car | 4,123 | Motor | 567 |
| Bus | 234 | Truck | 456 |
| Light | 1,234 | Sign | 890 |

---

## 🔧 Configuration

### Training Configuration (`config.yaml`)

```yaml
data:
  processed_dir: './processed_data'
  dataset_yaml: './processed_data/dataset.yaml'

model:
  type: 'yolov8'
  size: 's'              # n, s, m, l, x
  num_classes: 15
  input_channels: 3

training:
  epochs: 100
  batch_size: 16         # Reduce if GPU memory issues
  img_size: 640
  device: 'cuda'         # or 'cpu'
  workers: 4
  patience: 50
  save_period: 10

optimizer:
  name: 'AdamW'
  lr0: 0.001
  momentum: 0.937
  weight_decay: 0.0005

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
```

### Model Sizes

| Size | Parameters | Speed | Accuracy |
|------|-----------|-------|----------|
| **n** | 3M | Fastest | Good |
| **s** | 11M | Fast | Better (Recommended) |
| **m** | 26M | Medium | Great |
| **l** | 44M | Slow | Excellent |
| **x** | 68M | Slowest | Best |

---

## 🎯 Advanced Features

### Multi-Modal Fusion

Process RGB and Thermal images together:

```python
from complete_pipeline import TargetRecognitionPipeline

pipeline = TargetRecognitionPipeline(
    detector_path='runs/detect/multimodal_yolo/weights/best.pt',
    use_inpainting=True
)

results = pipeline.process_frame(
    rgb_image=rgb_img,
    thermal_image=thermal_img,
    conf_threshold=0.25
)
```

### Custom Training

Train on custom dataset:

```python
from train_pipeline import TrainingPipeline

pipeline = TrainingPipeline(config_path='custom_config.yaml')
model_path, metrics = pipeline.run()
```

### Export Models

Export for deployment:

```python
from detection_model import MultiModalYOLO

detector = MultiModalYOLO(model_size='s')
detector.load('runs/detect/multimodal_yolo/weights/best.pt')

# Export to ONNX
detector.export(format='onnx', imgsz=640)

# Export to TensorRT
detector.export(format='engine', imgsz=640)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```yaml
# In config.yaml, reduce batch size
training:
  batch_size: 4  # or even 2
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. GPU Not Detected

**Error**: `CUDA not available`

**Solution**:
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Inpainting Not Working

**Error**: `Inpainting unavailable`

**Solution**:
```bash
# Install optional dependencies
pip install diffusers transformers accelerate
```

#### 5. Streamlit Port Already in Use

**Error**: `Port 8501 is in use`

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Performance Optimization

**For faster inference:**
- Use smaller model size (n or s)
- Reduce image size to 416x416
- Disable inpainting
- Use FP16 precision on GPU

**For better accuracy:**
- Use larger model (m, l, or x)
- Train for more epochs (100+)
- Increase image size to 1280x1280
- Use data augmentation

---

## 📚 API Reference

### Detection Model

```python
from detection_model import MultiModalYOLO

# Initialize
detector = MultiModalYOLO(
    model_size='s',      # Model size
    num_classes=15,      # Number of classes
    input_channels=3     # Input channels
)

# Load model
detector.load('path/to/model.pt')

# Detect objects
results = detector.detect(
    image,               # Input image (numpy array)
    conf_threshold=0.25, # Confidence threshold
    iou_threshold=0.45   # NMS IoU threshold
)

# Visualize
vis_img = detector.visualize_results(
    image,               # Original image
    results,             # Detection results
    save_path='out.jpg', # Optional save path
    show_labels=True     # Show labels
)
```

### Inpainting Module

```python
from inpainting_module import TargetInpainter

# Initialize
inpainter = TargetInpainter(device='cuda')

# Reconstruct image
reconstructed, mask = inpainter.reconstruct_target(
    rgb_image,           # Input RGB image
    thermal_image=None,  # Optional thermal image
    detection_boxes=[], # Optional detection boxes
    object_type='car'    # Object type for better prompts
)
```

### Complete Pipeline

```python
from complete_pipeline import TargetRecognitionPipeline

# Initialize
pipeline = TargetRecognitionPipeline(
    detector_path='model.pt',
    use_inpainting=True,
    device='cuda'
)

# Process single frame
results = pipeline.process_frame(
    rgb_image,
    thermal_image=None,
    conf_threshold=0.25,
    enable_tracking=True
)

# Process video
pipeline.process_video(
    rgb_video_path='video.mp4',
    thermal_video_path=None,
    output_path='output.mp4',
    conf_threshold=0.25
)
```

---

## 🔬 Research & Development

### Future Enhancements

- [ ] Real-time video streaming support
- [ ] Multi-camera fusion
- [ ] Edge deployment (Jetson Nano, Raspberry Pi)
- [ ] Mobile app integration
- [ ] Cloud API deployment
- [ ] Advanced tracking algorithms (DeepSORT, ByteTrack)
- [ ] 3D object detection
- [ ] Semantic segmentation

### Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FLIR Systems** for the ADAS dataset
- **Ultralytics** for YOLOv8 implementation
- **Stability AI** for Stable Diffusion
- **Streamlit** for the web framework
- **PyTorch** community

---

## 📧 Contact

**Project Maintainer**: Your Name
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

**Project Link**: [https://github.com/yourusername/flir-adas-target-recognition](https://github.com/yourusername/flir-adas-target-recognition)

---

## 📈 Citation

If you use this project in your research, please cite:

```bibtex
@misc{flir_adas_recognition_2025,
  title={FLIR ADAS Target Recognition with Generative Inpainting},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/flir-adas-target-recognition}
}
```

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for Autonomous Driving Research**