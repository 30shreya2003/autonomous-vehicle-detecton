"""
Test script for inpainting functionality
"""

import cv2
import numpy as np
from pathlib import Path
from inpainting_module import TargetInpainter

def test_inpainting():
    print("="*70)
    print("  FLIR ADAS Inpainting Test")
    print("="*70)
    
    # Initialize inpainter
    print("\n[1/4] Initializing inpainter...")
    inpainter = TargetInpainter(device='cuda')
    
    # Get a test image
    print("\n[2/4] Loading test image...")
    val_images = list(Path('processed_data/images/val').glob('*.jpg'))
    
    if not val_images:
        print("❌ No validation images found!")
        print("Make sure you have processed the dataset.")
        return
    
    test_img_path = val_images[0]
    print(f"  Using: {test_img_path.name}")
    
    # Load image
    image = cv2.imread(str(test_img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a test mask (simulate occlusion in center)
    print("\n[3/4] Creating test occlusion mask...")
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create rectangular occlusion in center
    center_x, center_y = w // 2, h // 2
    mask[center_y-50:center_y+50, center_x-100:center_x+100] = 255
    
    print(f"  Mask size: {mask.sum() / 255} pixels")
    
    # Perform inpainting
    print("\n[4/4] Testing inpainting...")
    try:
        reconstructed = inpainter.inpaint_image(
            image_rgb,
            mask,
            object_type='car',
            num_inference_steps=20  # Fast for testing
        )
        
        # Save results
        output_dir = Path('inpainting_test_results')
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'original.jpg'), 
                   cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / 'mask.jpg'), mask)
        cv2.imwrite(str(output_dir / 'reconstructed.jpg'), 
                   cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        
        # Create comparison
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        comparison = np.hstack([image_rgb, mask_3ch, reconstructed])
        cv2.imwrite(str(output_dir / 'comparison.jpg'), 
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        print("\n✓ Inpainting test complete!")
        print(f"  Results saved to: {output_dir}/")
        print(f"  - original.jpg")
        print(f"  - mask.jpg")
        print(f"  - reconstructed.jpg")
        print(f"  - comparison.jpg")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Inpainting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inpainting()