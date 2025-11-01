"""
Complete Training Pipeline - Optimized for FLIR ADAS Dataset
Orchestrates data preparation, model training, and evaluation
"""

import os
import shutil
from pathlib import Path
import yaml
import json
from datetime import datetime

class TrainingPipeline:
    def __init__(self, config_path='config.yaml'):
        """Initialize training pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.setup_directories()
    
    def load_config(self, config_path):
        """Load training configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration optimized for FLIR ADAS
            config = {
                'data': {
                    'processed_dir': './processed_data',  # After setup_flir_dataset.py
                    'dataset_yaml': './processed_data/dataset.yaml'
                },
                'model': {
                    'type': 'yolov8',
                    'size': 's',  # n=fastest, s=balanced, m=accurate, l/x=best
                    'num_classes': 15,  # FLIR ADAS has 15 classes
                    'input_channels': 3  # 3 for RGB or Thermal, 4 for RGBT fusion
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 16,  # Reduce to 8 or 4 if GPU memory issues
                    'img_size': 640,
                    'device': 'cuda',  # 'cuda' or 'cpu'
                    'workers': 4,
                    'patience': 50,  # Early stopping patience
                    'save_period': 10  # Save checkpoint every N epochs
                },
                'optimizer': {
                    'name': 'AdamW',
                    'lr0': 0.001,  # Initial learning rate
                    'momentum': 0.937,
                    'weight_decay': 0.0005
                },
                'augmentation': {
                    'hsv_h': 0.015,  # HSV-Hue augmentation
                    'hsv_s': 0.7,    # HSV-Saturation
                    'hsv_v': 0.4,    # HSV-Value
                    'degrees': 0.0,  # Rotation (+/- deg)
                    'translate': 0.1, # Translation (+/- fraction)
                    'scale': 0.5,    # Scale (+/- gain)
                    'flipud': 0.0,   # Vertical flip probability
                    'fliplr': 0.5,   # Horizontal flip probability
                    'mosaic': 1.0    # Mosaic augmentation probability
                }
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"✓ Created default config at {config_path}")
        
        return config
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'models',
            'results',
            'logs'
        ]
        
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def verify_dataset(self):
        """Verify FLIR processed dataset is ready"""
        print("\n[1/5] Verifying processed dataset...")
        
        dataset_yaml = Path(self.config['data']['dataset_yaml'])
        
        if not dataset_yaml.exists():
            print(f"❌ Dataset YAML not found: {dataset_yaml}")
            print("\nPlease run setup_flir_dataset.py first:")
            print("  python setup_flir_dataset.py --flir-dir /path/to/flir --fusion fusion")
            return False
        
        # Load and verify dataset.yaml
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in dataset_config:
                print(f"❌ Missing field in dataset.yaml: {field}")
                return False
        
        # Verify directories exist
        dataset_path = Path(dataset_config['path'])
        train_dir = dataset_path / 'images' / 'train'
        val_dir = dataset_path / 'images' / 'val'
        
        if not train_dir.exists():
            print(f"❌ Training directory not found: {train_dir}")
            return False
        
        if not val_dir.exists():
            print(f"❌ Validation directory not found: {val_dir}")
            return False
        
        # Count images
        train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
        val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
        
        print(f"✓ Dataset verified:")
        print(f"  Path: {dataset_path}")
        print(f"  Classes: {dataset_config['nc']} ({', '.join(dataset_config['names'][:3])}...)")
        print(f"  Train images: {len(train_images)}")
        print(f"  Val images: {len(val_images)}")
        
        if len(train_images) == 0:
            print("❌ No training images found!")
            return False
        
        if len(val_images) == 0:
            print("⚠️  Warning: No validation images found")
        
        return True
    
    def train_model(self):
        """Train YOLOv8 detector on FLIR dataset"""
        print("\n[2/5] Initializing model and training...")
        
        from detection_model import MultiModalYOLO
        
        # Initialize detector
        detector = MultiModalYOLO(
            model_size=self.config['model']['size'],
            num_classes=self.config['model']['num_classes'],
            input_channels=self.config['model']['input_channels']
        )
        
        # Create model
        print(f"  Creating YOLOv8{self.config['model']['size']} model...")
        detector.create_custom_model()
        
        # Dataset YAML path
        dataset_yaml = str(Path(self.config['data']['dataset_yaml']).absolute())
        
        print(f"  Training configuration:")
        print(f"    Model: YOLOv8{self.config['model']['size']}")
        print(f"    Epochs: {self.config['training']['epochs']}")
        print(f"    Batch size: {self.config['training']['batch_size']}")
        print(f"    Image size: {self.config['training']['img_size']}")
        print(f"    Device: {self.config['training']['device']}")
        print(f"    Dataset: {dataset_yaml}")
        print("")
        
        # Train
        print("  Starting training... (this may take 30-60 minutes)")
        print("  " + "="*60)
        
        results = detector.train(
            data_yaml=dataset_yaml,
            epochs=self.config['training']['epochs'],
            imgsz=self.config['training']['img_size'],
            batch=self.config['training']['batch_size'],
            device=self.config['training']['device']
        )
        
        print("  " + "="*60)
        print("✓ Training complete!")
        
        return detector, results
    
    def evaluate_model(self, detector):
        """Evaluate trained model"""
        print("\n[3/5] Evaluating model on validation set...")
        
        dataset_yaml = str(Path(self.config['data']['dataset_yaml']).absolute())
        
        metrics = detector.evaluate(dataset_yaml, split='val')
        
        # Extract metrics
        metrics_dict = {
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }
        
        print(f"  Evaluation Results:")
        print(f"    mAP@0.5:     {metrics_dict['map50']:.4f}")
        print(f"    mAP@0.5:0.95: {metrics_dict['map50_95']:.4f}")
        print(f"    Precision:   {metrics_dict['precision']:.4f}")
        print(f"    Recall:      {metrics_dict['recall']:.4f}")
        
        # Performance assessment
        if metrics_dict['map50'] >= 0.85:
            print(f"  🎉 Excellent performance!")
        elif metrics_dict['map50'] >= 0.75:
            print(f"  ✓ Good performance")
        elif metrics_dict['map50'] >= 0.65:
            print(f"  ⚠️  Acceptable performance - consider training longer")
        else:
            print(f"  ⚠️  Low performance - check data or train longer")
        
        return metrics_dict
    
    def save_results(self, detector, metrics_dict):
        """Save model and results"""
        print("\n[4/5] Saving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path('results') / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = results_dir / 'best_model.pt'
        
        # Find the best model from YOLO training
        yolo_best = Path('runs/detect/multimodal_yolo/weights/best.pt')
        if yolo_best.exists():
            shutil.copy(yolo_best, model_path)
            print(f"  ✓ Model saved: {model_path}")
        else:
            print(f"  ⚠️  Warning: Could not find best.pt, saving current model")
            detector.save(str(model_path))
        
        # Save metrics
        metrics_file = results_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"  ✓ Metrics saved: {metrics_file}")
        
        # Save config used
        config_backup = results_dir / 'config.yaml'
        shutil.copy('config.yaml', config_backup)
        print(f"  ✓ Config saved: {config_backup}")
        
        # Create README with results
        readme_path = results_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("FLIR ADAS Target Recognition - Training Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training completed: {timestamp}\n\n")
            f.write("Model Configuration:\n")
            f.write(f"  Model: YOLOv8{self.config['model']['size']}\n")
            f.write(f"  Classes: {self.config['model']['num_classes']}\n")
            f.write(f"  Epochs: {self.config['training']['epochs']}\n")
            f.write(f"  Batch size: {self.config['training']['batch_size']}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  mAP@0.5:      {metrics_dict['map50']:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics_dict['map50_95']:.4f}\n")
            f.write(f"  Precision:    {metrics_dict['precision']:.4f}\n")
            f.write(f"  Recall:       {metrics_dict['recall']:.4f}\n\n")
            f.write("To use this model:\n")
            f.write(f"  python inference_demo.py --model {model_path} --mode webcam\n")
            f.write(f"  python inference_demo.py --model {model_path} --mode folder --input images/\n")
        
        print(f"  ✓ Summary saved: {readme_path}")
        
        return results_dir, model_path
    
    def generate_report(self, results_dir, model_path, metrics_dict):
        """Generate final training report"""
        print("\n[5/5] Generating training report...")
        
        print("\n" + "="*70)
        print("  TRAINING COMPLETE - FLIR ADAS DATASET")
        print("="*70)
        
        print(f"\n📁 Results Directory: {results_dir}")
        print(f"🎯 Model File: {model_path}")
        print(f"📊 Metrics File: {results_dir}/metrics.json")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  ├─ mAP@0.5:      {metrics_dict['map50']:.4f}")
        print(f"  ├─ mAP@0.5:0.95: {metrics_dict['map50_95']:.4f}")
        print(f"  ├─ Precision:    {metrics_dict['precision']:.4f}")
        print(f"  └─ Recall:       {metrics_dict['recall']:.4f}")
        
        print(f"\n🎓 Model Details:")
        print(f"  ├─ Architecture: YOLOv8{self.config['model']['size']}")
        print(f"  ├─ Classes: {self.config['model']['num_classes']} (FLIR ADAS)")
        print(f"  ├─ Input size: {self.config['training']['img_size']}x{self.config['training']['img_size']}")
        print(f"  └─ Trained epochs: {self.config['training']['epochs']}")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Test on webcam:")
        print(f"     python inference_demo.py --model {model_path} --mode webcam")
        print(f"")
        print(f"  2. Process validation images:")
        print(f"     python inference_demo.py --model {model_path} --mode folder \\")
        print(f"         --input processed_data/images/val --output results_val")
        print(f"")
        print(f"  3. Process video:")
        print(f"     python complete_pipeline.py --detector {model_path} \\")
        print(f"         --rgb-input video.mp4 --output output.mp4")
        print(f"")
        print(f"  4. With inpainting (for occluded targets):")
        print(f"     python complete_pipeline.py --detector {model_path} \\")
        print(f"         --rgb-input image.jpg --use-inpainting --output result.jpg")
        
        print("\n" + "="*70)
    
    def run(self):
        """Execute complete training pipeline"""
        print("="*70)
        print("  FLIR ADAS Target Recognition - Training Pipeline")
        print("="*70)
        
        # Verify dataset
        if not self.verify_dataset():
            print("\n❌ Dataset verification failed!")
            print("\nPlease ensure you have run:")
            print("  python setup_flir_dataset.py --flir-dir /path/to/flir --fusion fusion")
            return None, None
        
        # Train model
        detector, training_results = self.train_model()
        
        # Evaluate model
        metrics_dict = self.evaluate_model(detector)
        
        # Save results
        results_dir, model_path = self.save_results(detector, metrics_dict)
        
        # Generate report
        self.generate_report(results_dir, model_path, metrics_dict)
        
        return model_path, metrics_dict


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 on FLIR ADAS Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train_pipeline.py
  
  # Train with custom config
  python train_pipeline.py --config my_config.yaml
  
  # Quick test training (50 epochs, small model)
  # Edit config.yaml: model.size='n', epochs=50
  python train_pipeline.py
  
Before running, ensure you have processed the FLIR dataset:
  python setup_flir_dataset.py --flir-dir /path/to/flir --fusion fusion
        """
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Check if dataset is processed
    if not Path('processed_data/dataset.yaml').exists():
        print("\n" + "="*70)
        print("  FLIR Dataset Not Found")
        print("="*70)
        print("\n⚠️  Please process your FLIR dataset first:\n")
        print("  Step 1: Verify your dataset")
        print("    python verify_flir_dataset.py /path/to/your/flir/directory")
        print("")
        print("  Step 2: Process the dataset")
        print("    python setup_flir_dataset.py \\")
        print("        --flir-dir /path/to/your/flir/directory \\")
        print("        --output ./processed_data \\")
        print("        --fusion fusion")
        print("")
        print("  Step 3: Train (this script)")
        print("    python train_pipeline.py")
        print("\n" + "="*70)
        return
    
    # Run pipeline
    pipeline = TrainingPipeline(config_path=args.config)
    model_path, metrics = pipeline.run()
    
    if model_path:
        print(f"\n✓ Success! Model saved to: {model_path}")
    else:
        print(f"\n❌ Training failed. Check errors above.")
        exit(1)


if __name__ == "__main__":
    main()