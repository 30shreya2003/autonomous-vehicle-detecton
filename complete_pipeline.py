"""
Complete Multi-Modal Target Recognition Pipeline
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time
from typing import List, Dict, Optional

from detection_model import MultiModalYOLO
from inpainting_module import TargetInpainter

class TargetRecognitionPipeline:
    def __init__(self, detector_path: str, use_inpainting: bool = False, 
                 inpainting_model: str = "runwayml/stable-diffusion-inpainting",
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_inpainting = use_inpainting
        
        print("="*70)
        print("  Initializing Target Recognition Pipeline")
        print("="*70)
        
        print("\n[1/2] Loading detection model...")
        self.detector = MultiModalYOLO()
        self.detector.load(detector_path)
        print(f"  ✓ Detector loaded from {detector_path}")
        
        self.inpainter = None
        if use_inpainting:
            print("\n[2/2] Loading inpainting model...")
            try:
                self.inpainter = TargetInpainter(model_name=inpainting_model, device=self.device)
                print("  ✓ Inpainting enabled")
            except Exception as e:
                print(f"  ⚠️  Failed to load inpainter: {e}")
                self.use_inpainting = False
        
        self.tracked_objects = {}
        self.next_id = 0
        
        print("\n✓ Pipeline ready!")
        print("="*70 + "\n")
    
    def process_frame(self, rgb_image: np.ndarray, thermal_image: Optional[np.ndarray] = None,
                     conf_threshold: float = 0.25, enable_tracking: bool = True) -> Dict:
        start_time = time.time()
        
        results = {
            "original_image": rgb_image.copy(),
            "thermal_image": thermal_image.copy() if thermal_image is not None else None,
            "reconstructed_image": None,
            "mask": None,
            "detections": [],
            "tracked_objects": [],
            "metrics": {}
        }
        
        initial_detections = self.detector.detect(rgb_image, conf_threshold=conf_threshold)
        initial_boxes = initial_detections.boxes
        
        results["metrics"]["initial_detections"] = len(initial_boxes)
        
        boxes = initial_detections.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = self.detector.class_names[cls] if cls < len(self.detector.class_names) else f"class_{cls}"
            
            detection = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class": cls,
                "class_name": class_name,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2]
            }
            results["detections"].append(detection)
        
        results["metrics"]["processing_time"] = time.time() - start_time
        results["metrics"]["fps"] = 1.0 / results["metrics"]["processing_time"]
        
        return results
    
    def visualize_results(self, results: Dict, save_path: str = None, show_tracking: bool = False):
        img = results["original_image"].copy()
        
        for det in results["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det.get("class_name", "unknown")
            
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(img, f"Detections: {len(results['detections'])}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return img

if __name__ == "__main__":
    print("Complete Pipeline - Use with streamlit app or inference_demo.py")
