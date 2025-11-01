"""
FLIR ADAS Dataset Setup Script - FIXED VERSION
Handles the specific FLIR ADAS dataset structure and creates YOLO-compatible format
"""

import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

class FLIRDatasetSetup:
    def __init__(self, flir_root_dir, output_dir='./data/flir_processed'):
        """
        Initialize FLIR dataset processor
        
        Args:
            flir_root_dir: Root directory containing FLIR ADAS dataset
                          Should have: images_thermal_train, images_thermal_val, 
                                      images_rgb_train, images_rgb_val
            output_dir: Where to save processed dataset
        """
        self.flir_root = Path(flir_root_dir)
        self.output_dir = Path(output_dir)
        
        # Define FLIR structure
        self.splits = {
            'train': {
                'rgb': self.flir_root / 'images_rgb_train',
                'thermal': self.flir_root / 'images_thermal_train'
            },
            'val': {
                'rgb': self.flir_root / 'images_rgb_val',
                'thermal': self.flir_root / 'images_thermal_val'
            }
        }
        
        # FLIR ADAS classes (update based on your needs)
        self.class_map = {
            'person': 0,
            'bike': 1,
            'car': 2,
            'motor': 3,
            'bus': 4,
            'train': 5,
            'truck': 6,
            'light': 7,
            'hydrant': 8,
            'sign': 9,
            'dog': 10,
            'skateboard': 11,
            'stroller': 12,
            'scooter': 13,
            'other vehicle': 14
        }
        
        print(f"FLIR Dataset Root: {self.flir_root}")
        print(f"Output Directory: {self.output_dir}")
    
    def verify_structure(self):
        """Verify FLIR dataset structure exists"""
        print("\n[1/6] Verifying FLIR dataset structure...")
        
        missing = []
        for split, paths in self.splits.items():
            for modality, path in paths.items():
                if not path.exists():
                    missing.append(f"{split}/{modality}: {path}")
                else:
                    # Count images
                    images = list(path.glob('**/*.jpg')) + list(path.glob('**/*.jpeg'))
                    print(f"  ✓ {split}/{modality}: {len(images)} images")
        
        if missing:
            print("\n❌ Missing directories:")
            for m in missing:
                print(f"  • {m}")
            return False
        
        print("✓ Dataset structure verified")
        return True
    
    def parse_flir_annotations(self, annotation_file):
        """
        Parse FLIR COCO-style annotations
        Note: FLIR provides annotations in COCO JSON format
        """
        if not annotation_file.exists():
            return None, None
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to annotations mapping
        annotations_by_image = {}
        
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Create filename to annotations mapping
        filename_to_anns = {}
        for img in coco_data.get('images', []):
            image_id = img['id']
            filename = img['file_name']
            if image_id in annotations_by_image:
                filename_to_anns[filename] = {
                    'annotations': annotations_by_image[image_id],
                    'width': img['width'],
                    'height': img['height']
                }
        
        # Category ID to name mapping
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        
        return filename_to_anns, categories
    
    def coco_to_yolo(self, bbox, img_width, img_height):
        """
        Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
        All normalized to [0, 1]
        """
        x, y, w, h = bbox
        
        # Convert to center coordinates
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return x_center, y_center, width, height
    
    def setup_yolo_structure(self):
        """Create YOLO-compatible directory structure"""
        print("\n[2/6] Creating YOLO directory structure...")
        
        # Create directories
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                (self.output_dir / subdir / split).mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
    
    def process_split(self, split='train', use_thermal=False, fusion_method='rgb_only'):
        """
        Process a single split (train or val)
        
        Args:
            split: 'train' or 'val'
            use_thermal: Whether to use thermal images
            fusion_method: 'rgb_only', 'thermal_only', or 'fusion'
        """
        print(f"\n[3/6] Processing {split} split (method: {fusion_method})...")
        
        rgb_dir = self.splits[split]['rgb']
        thermal_dir = self.splits[split]['thermal']
        
        # FIXED: Look for coco.json in the RGB directory
        annotation_file = rgb_dir / 'coco.json'
        
        # Parse annotations if available
        annotations = None
        categories = None
        if annotation_file.exists():
            print(f"  ✓ Loading annotations from {annotation_file}")
            annotations, categories = self.parse_flir_annotations(annotation_file)
            if annotations:
                print(f"    Found annotations for {len(annotations)} images")
        else:
            print(f"  ⚠ No annotations found at {annotation_file}")
            print(f"  Will create dummy labels (you'll need to provide annotations)")
        
        # Get all RGB images
        rgb_images = sorted(list(rgb_dir.glob('**/*.jpg')) + list(rgb_dir.glob('**/*.jpeg')))
        
        # Filter out the coco.json from image list if accidentally included
        rgb_images = [img for img in rgb_images if img.suffix.lower() in ['.jpg', '.jpeg']]
        
        if len(rgb_images) == 0:
            print(f"  ⚠ No images found in {rgb_dir}")
            return
        
        print(f"  Processing {len(rgb_images)} images...")
        
        processed_count = 0
        for rgb_path in tqdm(rgb_images, desc=f"  {split}"):
            try:
                # Load RGB image
                rgb_img = cv2.imread(str(rgb_path))
                if rgb_img is None:
                    continue
                
                # Find corresponding thermal image
                # FLIR naming: RGB might be in 'data' subfolder, thermal in 'PreviewData'
                thermal_path = None
                relative_path = rgb_path.relative_to(rgb_dir)
                
                # Try direct match
                potential_thermal = thermal_dir / relative_path
                if potential_thermal.exists():
                    thermal_path = potential_thermal
                else:
                    # Try without 'data' subfolder (thermal might be in root)
                    thermal_path = thermal_dir / relative_path.name
                
                # Process based on fusion method
                if fusion_method == 'rgb_only':
                    final_img = rgb_img
                elif fusion_method == 'thermal_only' and thermal_path and thermal_path.exists():
                    thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
                    final_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2BGR)
                elif fusion_method == 'fusion' and thermal_path and thermal_path.exists():
                    # Create RGB + Thermal fusion
                    thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
                    
                    # Resize if needed
                    if thermal_img.shape[:2] != rgb_img.shape[:2]:
                        thermal_img = cv2.resize(thermal_img, 
                                                (rgb_img.shape[1], rgb_img.shape[0]))
                    
                    # Create overlay fusion
                    thermal_colored = cv2.applyColorMap(thermal_img, cv2.COLORMAP_JET)
                    final_img = cv2.addWeighted(rgb_img, 0.6, thermal_colored, 0.4, 0)
                else:
                    final_img = rgb_img
                
                # Generate output filename
                output_name = rgb_path.stem + '.jpg'
                output_img_path = self.output_dir / 'images' / split / output_name
                
                # Save image
                cv2.imwrite(str(output_img_path), final_img)
                
                # Process labels
                label_path = self.output_dir / 'labels' / split / (rgb_path.stem + '.txt')
                
                # FIXED: Try multiple filename formats
                found_annotation = False
                if annotations:
                    # Try different filename formats
                    possible_names = [
                        rgb_path.name,  # Just filename
                        str(relative_path).replace('\\', '/'),  # Relative path with forward slashes
                        str(relative_path),  # Relative path
                    ]
                    
                    for name in possible_names:
                        if name in annotations:
                            # Convert COCO annotations to YOLO format
                            ann_data = annotations[name]
                            img_width = ann_data['width']
                            img_height = ann_data['height']
                            
                            with open(label_path, 'w') as f:
                                for ann in ann_data['annotations']:
                                    category_id = ann['category_id']
                                    category_name = categories.get(category_id, 'unknown')
                                    
                                    # Map to class index
                                    if category_name in self.class_map:
                                        class_idx = self.class_map[category_name]
                                    else:
                                        class_idx = 0  # Default to person
                                    
                                    # Convert bbox
                                    bbox = ann['bbox']
                                    x_center, y_center, width, height = self.coco_to_yolo(
                                        bbox, img_width, img_height
                                    )
                                    
                                    # Write YOLO format line
                                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            
                            found_annotation = True
                            break
                
                if not found_annotation:
                    # Create empty label file (no annotations)
                    label_path.touch()
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n  ✗ Error processing {rgb_path.name}: {e}")
                continue
        
        print(f"  ✓ Processed {processed_count} images")
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration"""
        print("\n[4/6] Creating dataset.yaml...")
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_map),
            'names': list(self.class_map.keys())
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"  ✓ Created {yaml_path}")
        return yaml_path
    
    def create_statistics(self):
        """Generate dataset statistics"""
        print("\n[5/6] Generating statistics...")
        
        stats = {
            'train': {'images': 0, 'labels': 0, 'labels_with_objects': 0},
            'val': {'images': 0, 'labels': 0, 'labels_with_objects': 0}
        }
        
        for split in ['train', 'val']:
            img_dir = self.output_dir / 'images' / split
            lbl_dir = self.output_dir / 'labels' / split
            
            if img_dir.exists():
                stats[split]['images'] = len(list(img_dir.glob('*.jpg')))
            if lbl_dir.exists():
                label_files = list(lbl_dir.glob('*.txt'))
                stats[split]['labels'] = len(label_files)
                
                # Count non-empty labels
                for lbl_file in label_files:
                    if lbl_file.stat().st_size > 0:
                        stats[split]['labels_with_objects'] += 1
        
        # Save statistics
        stats_file = self.output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n  Dataset Statistics:")
        print(f"    Train: {stats['train']['images']} images, {stats['train']['labels_with_objects']}/{stats['train']['labels']} labels with objects")
        print(f"    Val:   {stats['val']['images']} images, {stats['val']['labels_with_objects']}/{stats['val']['labels']} labels with objects")
        
        return stats
    
    def run(self, fusion_method='rgb_only'):
        """
        Execute complete FLIR dataset setup
        
        Args:
            fusion_method: 'rgb_only', 'thermal_only', or 'fusion'
        """
        print("="*70)
        print("  FLIR ADAS Dataset Setup")
        print("="*70)
        
        # Verify structure
        if not self.verify_structure():
            print("\n❌ Dataset structure verification failed!")
            print("\nExpected structure:")
            print("  flir_root/")
            print("    ├── images_rgb_train/")
            print("    ├── images_rgb_val/")
            print("    ├── images_thermal_train/")
            print("    └── images_thermal_val/")
            return False
        
        # Setup YOLO structure
        self.setup_yolo_structure()
        
        # Process splits
        for split in ['train', 'val']:
            self.process_split(split, fusion_method=fusion_method)
        
        # Create dataset yaml
        yaml_path = self.create_dataset_yaml()
        
        # Generate statistics
        stats = self.create_statistics()
        
        print("\n[6/6] Setup complete!")
        print("="*70)
        print(f"\n✓ Dataset ready at: {self.output_dir}")
        print(f"✓ Config file: {yaml_path}")
        print(f"\nNext steps:")
        print(f"  1. Verify data: ls {self.output_dir}/images/train/")
        print(f"  2. Train model: python train_pipeline.py --config config.yaml")
        print("="*70)
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup FLIR ADAS Dataset')
    parser.add_argument('--flir-dir', type=str, required=True,
                       help='Path to FLIR ADAS root directory')
    parser.add_argument('--output', type=str, default='./data/flir_processed',
                       help='Output directory for processed dataset')
    parser.add_argument('--fusion', type=str, default='rgb_only',
                       choices=['rgb_only', 'thermal_only', 'fusion'],
                       help='Image processing method')
    
    args = parser.parse_args()
    
    # Initialize and run
    setup = FLIRDatasetSetup(args.flir_dir, args.output)
    success = setup.run(fusion_method=args.fusion)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()