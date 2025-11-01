"""
Comprehensive Testing and Validation Script - FLIR ADAS Optimized
Tests all components of the pipeline
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import cv2
import torch

class PipelineTester:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def test_imports(self):
        """Test all required imports"""
        print("\n[TEST 1/10] Testing imports...")
        
        required_modules = {
            'torch': 'PyTorch',
            'cv2': 'OpenCV',
            'numpy': 'NumPy',
            'PIL': 'Pillow',
            'ultralytics': 'YOLOv8',
            'yaml': 'PyYAML',
            'pathlib': 'Pathlib'
        }
        
        for module, name in required_modules.items():
            try:
                __import__(module)
                self.results['passed'].append(f"Import: {name}")
                print(f"  ✓ {name}")
            except ImportError as e:
                self.results['failed'].append(f"Import: {name} - {e}")
                print(f"  ✗ {name}: {e}")
        
        # Optional imports (for inpainting)
        optional_modules = {
            'diffusers': 'Diffusers (for inpainting)',
            'transformers': 'Transformers (for inpainting)'
        }
        
        for module, name in optional_modules.items():
            try:
                __import__(module)
                self.results['passed'].append(f"Import (optional): {name}")
                print(f"  ✓ {name}")
            except ImportError:
                self.results['warnings'].append(f"Optional: {name} not installed (inpainting disabled)")
                print(f"  ⚠ {name} not installed (optional)")
    
    def test_gpu(self):
        """Test GPU availability"""
        print("\n[TEST 2/10] Testing GPU...")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.results['passed'].append(f"GPU: {gpu_name} ({memory:.1f}GB)")
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"  ✓ Memory: {memory:.1f}GB")
            
            # Test CUDA
            try:
                x = torch.rand(100, 100).cuda()
                y = x @ x
                self.results['passed'].append("CUDA operations")
                print(f"  ✓ CUDA operations working")
            except Exception as e:
                self.results['failed'].append(f"CUDA operations: {e}")
                print(f"  ✗ CUDA error: {e}")
        else:
            self.results['warnings'].append("GPU: Not available (will use CPU)")
            print("  ⚠ GPU not available - will use CPU (slower)")
    
    def test_flir_dataset(self):
        """Test FLIR dataset setup"""
        print("\n[TEST 3/10] Testing FLIR dataset...")
        
        # FIXED: Check for processed_data instead of data/flir_processed
        flir_processed = Path('processed_data')
        
        if flir_processed.exists():
            # Check for dataset.yaml
            dataset_yaml = flir_processed / 'dataset.yaml'
            if dataset_yaml.exists():
                self.results['passed'].append("FLIR dataset: Processed")
                print(f"  ✓ Processed FLIR dataset found at: {flir_processed}")
                
                # Check images
                train_images = flir_processed / 'images' / 'train'
                val_images = flir_processed / 'images' / 'val'
                
                if train_images.exists():
                    train_count = len(list(train_images.glob('*.jpg')))
                    print(f"  ✓ Train images: {train_count}")
                    self.results['passed'].append(f"Train images: {train_count}")
                else:
                    self.results['warnings'].append("Train images not found")
                    print(f"  ⚠ Train images not found")
                
                if val_images.exists():
                    val_count = len(list(val_images.glob('*.jpg')))
                    print(f"  ✓ Val images: {val_count}")
                    self.results['passed'].append(f"Val images: {val_count}")
                else:
                    self.results['warnings'].append("Val images not found")
                    print(f"  ⚠ Val images not found")
            else:
                self.results['warnings'].append("FLIR dataset: Exists but not fully processed")
                print(f"  ⚠ dataset.yaml not found - run setup_flir_dataset.py")
        else:
            self.results['warnings'].append("FLIR dataset: Not found")
            print(f"  ⚠ FLIR dataset not processed yet")
            print(f"    Run: python setup_flir_dataset.py --flir-dir D:/ATR GenAI/data_dir/FLIR_ADAS_v2")
    
    def test_detection_model(self):
        """Test detection model module"""
        print("\n[TEST 4/10] Testing detection model...")
        
        try:
            from detection_model import MultiModalYOLO
            
            # Create detector
            detector = MultiModalYOLO(model_size='n', num_classes=15)
            self.results['passed'].append("Detection model: Initialization")
            print(f"  ✓ MultiModalYOLO initialized")
            
            # Check FLIR classes
            if len(detector.class_names) == 15:
                self.results['passed'].append("Detection model: FLIR classes")
                print(f"  ✓ FLIR ADAS classes loaded ({len(detector.class_names)} classes)")
            else:
                self.results['warnings'].append(f"Detection model: Expected 15 classes, got {len(detector.class_names)}")
            
            # Try to create model (lightweight test)
            try:
                detector.create_custom_model()
                self.results['passed'].append("Detection model: Model creation")
                print(f"  ✓ Model created successfully")
            except Exception as e:
                self.results['warnings'].append(f"Detection model creation: {e}")
                print(f"  ⚠ Model creation warning: {e}")
            
        except Exception as e:
            self.results['failed'].append(f"Detection model: {e}")
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
    
    def test_inpainting_module(self):
        """Test inpainting module (without loading model)"""
        print("\n[TEST 5/10] Testing inpainting module...")
        
        try:
            from inpainting_module import TargetInpainter
            
            # Initialize without loading model
            inpainter = TargetInpainter(device='cpu')
            
            self.results['passed'].append("Inpainting: Module structure")
            print(f"  ✓ TargetInpainter initialized")
            
            # Check FLIR prompts
            if 'person' in inpainter.flir_prompts and 'car' in inpainter.flir_prompts:
                self.results['passed'].append("Inpainting: FLIR prompts")
                print(f"  ✓ FLIR-specific prompts loaded")
            
            # Test mask creation (lightweight)
            try:
                test_image = np.random.randint(0, 255, (512, 640, 3), dtype=np.uint8)
                mask = inpainter.create_smart_mask(test_image)
                
                if mask.shape == (512, 640):
                    self.results['passed'].append("Inpainting: Mask creation")
                    print(f"  ✓ Mask creation works")
            except Exception as e:
                self.results['warnings'].append(f"Mask creation: {e}")
                print(f"  ⚠ Mask creation warning: {e}")
            
            print(f"  ℹ️  Note: Stable Diffusion model not tested (loads on first use)")
            
        except Exception as e:
            self.results['failed'].append(f"Inpainting module: {e}")
            print(f"  ✗ Error: {e}")
    
    def test_complete_pipeline(self):
        """Test complete pipeline structure"""
        print("\n[TEST 6/10] Testing complete pipeline...")
        
        try:
            from complete_pipeline import TargetRecognitionPipeline
            
            self.results['passed'].append("Pipeline: Module structure")
            print(f"  ✓ TargetRecognitionPipeline structure valid")
            
            # FIXED: Check multiple possible locations for trained models
            model_locations = [
                Path('results'),
                Path('runs/detect/multimodal_yolo/weights')
            ]
            
            models_found = []
            for loc in model_locations:
                if loc.exists():
                    if 'results' in str(loc):
                        models = list(loc.glob('*/best_model.pt'))
                    else:
                        models = list(loc.glob('best.pt')) + list(loc.glob('best_model.pt'))
                    models_found.extend(models)
            
            if models_found:
                print(f"  ✓ Found {len(models_found)} trained model(s)")
                for model in models_found:
                    print(f"    • {model}")
                self.results['passed'].append(f"Pipeline: {len(models_found)} trained model(s)")
            else:
                print(f"  ⚠ No trained models found")
                self.results['warnings'].append("Pipeline: No trained models (train first)")
            
        except Exception as e:
            self.results['failed'].append(f"Pipeline: {e}")
            print(f"  ✗ Error: {e}")
    
    def test_training_pipeline(self):
        """Test training pipeline structure"""
        print("\n[TEST 7/10] Testing training pipeline...")
        
        try:
            from train_pipeline import TrainingPipeline
            
            # Check config
            config_path = Path('config.yaml')
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.results['passed'].append("Training: Config file")
                print(f"  ✓ config.yaml exists")
                
                # Check config structure
                required_keys = ['data', 'model', 'training']
                for key in required_keys:
                    if key in config:
                        print(f"  ✓ Config section: {key}")
                    else:
                        self.results['warnings'].append(f"Config missing: {key}")
                        print(f"  ⚠ Config missing: {key}")
                
                # Check device setting
                if 'training' in config and 'device' in config['training']:
                    device = config['training']['device']
                    print(f"  ✓ Training device: {device}")
            else:
                self.results['warnings'].append("Training: No config file")
                print(f"  ⚠ config.yaml not found (will be created on first run)")
            
            self.results['passed'].append("Training: Pipeline structure")
            print(f"  ✓ TrainingPipeline structure valid")
            
        except Exception as e:
            self.results['failed'].append(f"Training pipeline: {e}")
            print(f"  ✗ Error: {e}")
    
    def test_inference_demo(self):
        """Test inference demo script"""
        print("\n[TEST 8/10] Testing inference demo...")
        
        # FIXED: Check if file exists first
        inference_demo_path = Path('inference_demo.py')
        if not inference_demo_path.exists():
            self.results['warnings'].append("Inference demo: Script not found")
            print(f"  ⚠ inference_demo.py not found (optional)")
            return
        
        try:
            from inference_demo import RealtimeDemo
            
            self.results['passed'].append("Inference: Demo structure")
            print(f"  ✓ RealtimeDemo structure valid")
            
        except Exception as e:
            self.results['warnings'].append(f"Inference demo: {e}")
            print(f"  ⚠ Inference demo: {e}")
    
    def test_flir_setup(self):
        """Test FLIR setup scripts"""
        print("\n[TEST 9/10] Testing FLIR setup scripts...")
        
        scripts = {
            'setup_flir_dataset.py': 'Required',
            'detection_model.py': 'Required',
            'train_pipeline.py': 'Required',
            'complete_pipeline.py': 'Required',
            'inpainting_module.py': 'Optional'
        }
        
        for script, status in scripts.items():
            if Path(script).exists():
                self.results['passed'].append(f"Script: {script}")
                print(f"  ✓ {script}")
            else:
                if status == 'Required':
                    self.results['failed'].append(f"Script: {script} missing")
                    print(f"  ✗ {script} missing (required)")
                else:
                    self.results['warnings'].append(f"Script: {script} missing")
                    print(f"  ⚠ {script} missing (optional)")
    
    def test_directory_structure(self):
        """Test directory structure"""
        print("\n[TEST 10/10] Testing directory structure...")
        
        dirs = {
            'processed_data': 'Processed dataset directory',
            'results': 'Results directory',
            'models': 'Models directory',
            'logs': 'Logs directory',
            'runs': 'Training runs directory'
        }
        
        for dir_name, description in dirs.items():
            dir_path = Path(dir_name)
            if dir_path.exists():
                self.results['passed'].append(f"Directory: {dir_name}")
                print(f"  ✓ {description}")
            else:
                self.results['warnings'].append(f"Directory: {dir_name} not found")
                print(f"  ⚠ {description} not found")
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*70)
        print("  FLIR ADAS PIPELINE - TESTING SUITE")
        print("="*70)
        
        self.test_imports()
        self.test_gpu()
        self.test_flir_dataset()
        self.test_detection_model()
        self.test_inpainting_module()
        self.test_complete_pipeline()
        self.test_training_pipeline()
        self.test_inference_demo()
        self.test_flir_setup()
        self.test_directory_structure()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70)
        
        print(f"\n✓ PASSED: {len(self.results['passed'])}")
        for test in self.results['passed'][:10]:  # Show first 10
            print(f"  • {test}")
        if len(self.results['passed']) > 10:
            print(f"  ... and {len(self.results['passed']) - 10} more")
        
        if self.results['warnings']:
            print(f"\n⚠ WARNINGS: {len(self.results['warnings'])}")
            for warning in self.results['warnings']:
                print(f"  • {warning}")
        
        if self.results['failed']:
            print(f"\n✗ FAILED: {len(self.results['failed'])}")
            for failure in self.results['failed']:
                print(f"  • {failure}")
        
        print("\n" + "="*70)
        
        # Determine status
        if self.results['failed']:
            print("❌ Some tests failed - please fix issues before proceeding")
            print("="*70)
            return False
        elif self.results['warnings']:
            print("⚠️  All critical tests passed - warnings are optional features")
            print("="*70)
            return True
        else:
            print("✅ All tests passed - ready to use!")
            print("="*70)
            return True


def main():
    tester = PipelineTester()
    success = tester.run_all_tests()
    
    print("\n📋 Next Steps:")
    
    if success:
        # Check what's available
        flir_processed = Path('processed_data/dataset.yaml')
        trained_model = (Path('runs/detect/multimodal_yolo/weights/best.pt').exists() or 
                        (Path('results').exists() and list(Path('results').glob('*/best_model.pt'))))
        
        if trained_model:
            print("\n✅ System Ready!")
            
            # Find model path
            if Path('runs/detect/multimodal_yolo/weights/best.pt').exists():
                model_path = 'runs/detect/multimodal_yolo/weights/best.pt'
            else:
                model_path = sorted(Path('results').glob('*/best_model.pt'))[-1]
            
            print(f"\n🎯 Your trained model: {model_path}")
            
            print("\n📸 Test your model:")
            print(f"   python quick_visual_test.py")
            
            print("\n🎥 Run demos:")
            print(f"   # Webcam")
            print(f"   python inference_demo.py --model {model_path} --mode webcam")
            print(f"\n   # Process images")
            print(f"   python inference_demo.py --model {model_path} --mode folder --input images/")
        else:
            print("\n⚠️  No trained model found")
            print("\n🎓 Train your model:")
            print("   python train_pipeline.py")
    else:
        print("\n1️⃣  Fix failed tests")
        print("2️⃣  Install missing dependencies:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   pip install ultralytics opencv-python numpy pyyaml tqdm")
        print("3️⃣  Re-run tests:")
        print("   python test_pipeline.py")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()