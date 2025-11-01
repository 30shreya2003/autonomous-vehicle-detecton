"""
Test inpainting integrated with detection
"""

import cv2
import numpy as np
from pathlib import Path
from detection_model import MultiModalYOLO
from inpainting_module import TargetInpainter

def test_full_pipeline():
    print("="*70)
    print("  FULL PIPELINE TEST: Detection + Inpainting")
    print("="*70)
    
    # Load models
    print("\n[1/4] Loading models...")
    detector = MultiModalYOLO(model_size='s')
    detector.load('runs/detect/multimodal_yolo/weights/best.pt')
    print("  ✓ Detector loaded")
    
    inpainter = TargetInpainter(device='cuda')
    print("  ✓ Inpainter loaded")
    
    # Get test image
    print("\n[2/4] Loading test image...")
    val_images = list(Path('processed_data/images/val').glob('*.jpg'))
    if not val_images:
        print("❌ No validation images found!")
        return
    
    test_img_path = val_images[0]
    print(f"  Using: {test_img_path.name}")
    
    image = cv2.imread(str(test_img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect
    print("\n[3/4] Running detection...")
    results = detector.detect(image_rgb, conf_threshold=0.25)
    print(f"  Detections: {len(results.boxes)}")
    
    # Get detection boxes
    detection_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detection_boxes.append([x1, y1, x2, y2])
        print(f"    Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Create mask with VERY OBVIOUS occlusion
    print("\n[4/4] Creating artificial occlusion and inpainting...")
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create large black rectangle occlusion in center
    center_x, center_y = w // 2, h // 2
    mask[center_y-100:center_y+100, center_x-150:center_x+150] = 255
    
    # Also black out that region in the image
    image_occluded = image_rgb.copy()
    image_occluded[center_y-100:center_y+100, center_x-150:center_x+150] = [0, 0, 0]
    
    print(f"  Created {mask.sum() / 255:.0f} pixel occlusion")
    
    # Inpaint
    print("  Running inpainting (this takes 15-30 seconds)...")
    reconstructed = inpainter.inpaint_image(
        image_occluded,
        mask,
        object_type='car',
        num_inference_steps=30
    )
    
    # Save results
    output_dir = Path('full_pipeline_test')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / '1_original.jpg'), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / '2_occluded.jpg'), cv2.cvtColor(image_occluded, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / '3_mask.jpg'), mask)
    cv2.imwrite(str(output_dir / '4_reconstructed.jpg'), cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
    
    # Create comparison
    comparison = np.hstack([image_occluded, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), reconstructed])
    cv2.imwrite(str(output_dir / '5_comparison.jpg'), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"  Check if the black rectangle was filled in!")
    print("="*70)

if __name__ == "__main__":
    test_full_pipeline()