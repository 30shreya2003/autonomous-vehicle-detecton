from detection_model import MultiModalYOLO
import cv2
from pathlib import Path

if __name__ == '__main__':
    print("Loading model...")
    detector = MultiModalYOLO(model_size='s')
    detector.load('runs/detect/multimodal_yolo/weights/best.pt')

    print("\nTesting on 3 sample images...")
    val_images = list(Path('processed_data/images/val').glob('*.jpg'))[:3]
    
    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)
    
    total_dets = 0
    for i, img_path in enumerate(val_images):
        print(f"\n[{i+1}/3] {img_path.name}")
        
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = detector.detect(image_rgb, conf_threshold=0.25)
        vis = detector.visualize_results(image_rgb, results, 
                                        save_path=str(results_dir / f'result_{i+1}.jpg'))
        
        print(f"  Detections: {len(results.boxes)}")
        total_dets += len(results.boxes)
        
        for box in results.boxes[:3]:  # Show first 3
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    - {detector.class_names[cls]}: {conf:.2f}")
    
    print(f"\n✓ Total detections: {total_dets}")
    print(f"✓ Results saved to: test_results/")
    print(f"\nView results: explorer test_results")
