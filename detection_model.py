"""
Multi-Modal Object Detection using YOLOv8 - Optimized for FLIR ADAS
Supports RGB, Thermal, and Fused inputs
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml

class MultiModalYOLO:
    def __init__(self, model_size='s', num_classes=15, input_channels=3):
        """
        Initialize Multi-Modal YOLO detector for FLIR ADAS
        
        Args:
            model_size: 'n' (fastest), 's' (balanced), 'm' (accurate), 'l', 'x' (best)
            num_classes: 15 for FLIR ADAS (person, bike, car, motor, bus, etc.)
            input_channels: 3 for RGB/Thermal, 4 for RGBT fusion
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.model = None
        
        # FLIR ADAS class names
        self.class_names = [
            'person', 'bike', 'car', 'motor', 'bus', 
            'train', 'truck', 'light', 'hydrant', 'sign',
            'dog', 'skateboard', 'stroller', 'scooter', 'other vehicle'
        ]
        
        print(f"Initialized YOLOv8{model_size} for FLIR ADAS ({num_classes} classes)")
    
    def create_custom_model(self):
        """
        Create YOLOv8 model
        For FLIR: Uses standard 3-channel input (RGB or Thermal or Fusion overlay)
        """
        # Start with pretrained model
        base_model = f'yolov8{self.model_size}.pt'
        self.model = YOLO(base_model)
        
        print(f"✓ Loaded YOLOv8{self.model_size} pretrained weights")
        
        # Note: For 4-channel RGBT fusion, would need to modify first conv layer
        # For FLIR, we use 3-channel fusion (RGB+Thermal overlay) which works directly
        if self.input_channels == 4:
            print("⚠️  Warning: 4-channel input requires model modification")
            print("   Using 3-channel fusion (overlay) instead for better compatibility")
            self.input_channels = 3
        
        return self.model
    
    def train(self, data_yaml, epochs=100, imgsz=640, batch=16, device='cuda'):
        """
        Train the detection model on FLIR ADAS
        
        Args:
            data_yaml: Path to dataset.yaml (from setup_flir_dataset.py)
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size (reduce if GPU memory issues)
            device: 'cuda' or 'cpu'
        """
        if self.model is None:
            self.create_custom_model()
        
        print(f"\nTraining YOLOv8{self.model_size} on FLIR ADAS dataset")
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
        print(f"Device: {device}")
        print("="*60)
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='runs/detect',
            name='multimodal_yolo',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,  # Final learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # Box loss gain
            cls=0.5,  # Classification loss gain
            dfl=1.5,  # Distribution focal loss gain
            patience=50,  # Early stopping patience
            save=True,
            save_period=10,  # Save checkpoint every N epochs
            plots=True,
            verbose=True
        )
        
        print("="*60)
        print("✓ Training completed!")
        
        return results
    
    def detect(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run detection on single image
        
        Args:
            image: numpy array (H, W, C) - RGB or Thermal or Fusion
            conf_threshold: Confidence threshold (lower = more detections)
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Detection results with boxes, confidences, classes
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load or train a model first.")
        
        # Ensure 3-channel input
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # 4-channel to 3-channel (shouldn't happen with our setup)
            image = image[:, :, :3]
        
        # Run detection
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.model.device
        )
        
        return results[0]
    
    def visualize_results(self, image, results, save_path=None, show_labels=True):
        """
        Visualize detection results on image
        
        Args:
            image: Input image
            results: Detection results from detect()
            save_path: Path to save visualization
            show_labels: Show class labels and confidence
        
        Returns:
            Annotated image
        """
        # Create copy for visualization
        vis_image = image.copy()
        
        # Get detections
        boxes = results.boxes
        
        if len(boxes) == 0:
            if save_path:
                cv2.imwrite(save_path, vis_image)
            return vis_image
        
        # Draw each detection
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
            
            # Color based on class
            colors = {
                'person': (0, 255, 0),      # Green
                'car': (255, 0, 0),         # Blue
                'bike': (0, 255, 255),      # Yellow
                'motor': (255, 255, 0),     # Cyan
                'bus': (255, 0, 255),       # Magenta
                'truck': (128, 0, 255),     # Purple
            }
            color = colors.get(class_name, (0, 165, 255))  # Orange default
            
            # Draw bounding box
            thickness = 2 if conf > 0.5 else 1
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            if show_labels:
                # Prepare label
                label = f"{class_name}: {conf:.2f}"
                
                # Get label size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
        
        # Add detection count
        cv2.putText(
            vis_image,
            f"Detections: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def evaluate(self, data_yaml, split='val'):
        """
        Evaluate model on validation/test set
        
        Args:
            data_yaml: Path to dataset.yaml
            split: 'val' or 'test'
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        print(f"\nEvaluating on {split} set...")
        
        metrics = self.model.val(
            data=data_yaml,
            split=split,
            batch=16,
            plots=True,
            verbose=True
        )
        
        print(f"\nEvaluation Results:")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        
        return metrics
    
    def save(self, path):
        """Save model"""
        if self.model is not None:
            # Save to specified path
            self.model.save(path)
            print(f"✓ Model saved to {path}")
        else:
            print("❌ No model to save")
    
    def load(self, path):
        """Load model"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = YOLO(path)
        print(f"✓ Model loaded from {path}")
        
        # Try to infer model size from path or model
        try:
            model_name = Path(path).stem
            if 'yolov8' in model_name:
                self.model_size = model_name.split('yolov8')[1][0]
        except:
            pass
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
        
        info = {
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'class_names': self.class_names,
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
        }
        
        return info
    
    def export(self, format='onnx', imgsz=640):
        """
        Export model to different formats for deployment
        
        Args:
            format: 'onnx', 'torchscript', 'tflite', 'edgetpu', 'coreml', etc.
            imgsz: Input image size
        
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        print(f"Exporting model to {format} format...")
        
        export_path = self.model.export(
            format=format,
            imgsz=imgsz,
            dynamic=False,
            simplify=True
        )
        
        print(f"✓ Model exported to: {export_path}")
        
        return export_path


if __name__ == "__main__":
    print("This module provides MultiModalYOLO class for FLIR ADAS detection.")
    print("For training, use: python train_pipeline.py")
    print("For detection, use: python inference_demo.py")
