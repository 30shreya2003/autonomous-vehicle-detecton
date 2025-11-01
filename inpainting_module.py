"""
Generative Inpainting Module for Target Reconstruction - Optimized for FLIR ADAS
Uses Stable Diffusion to reconstruct occluded or degraded targets
Specialized for automotive/pedestrian detection scenarios
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class TargetInpainter:
    def __init__(self, model_name="runwayml/stable-diffusion-inpainting", device="cuda"):
        """
        Initialize inpainting model for FLIR ADAS target reconstruction
        
        Args:
            model_name: HuggingFace model ID
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_loaded = False
        
        print(f"Initializing inpainting module (device: {self.device})")
        
        # FLIR-specific prompts for different object types
        self.flir_prompts = {
            'person': "person, pedestrian, human, clear visibility, high detail, realistic, street scene",
            'car': "car, vehicle, automobile, clear view, high detail, realistic, road scene",
            'bike': "bicycle, bike, cyclist, clear visibility, detailed, realistic",
            'motor': "motorcycle, motorbike, clear view, detailed, realistic",
            'bus': "bus, large vehicle, clear visibility, detailed, realistic",
            'truck': "truck, large vehicle, clear view, detailed, realistic",
            'default': "vehicle or person, clear visibility, high detail, realistic, street scene"
        }
        
        print("✓ Inpainting module initialized (model will load on first use)")
    
    def _load_model(self):
        """Load with aggressive memory optimization for 4GB GPU"""
        if self.model_loaded:
            return
        
        print("\n" + "="*70)
        print("Loading Stable Diffusion (4GB GPU optimized)...")
        print("="*70)
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch
            
            # Use CPU instead of GPU for inpainting (saves VRAM)
            print("⚠️  Using CPU for inpainting (slower but works with limited VRAM)")
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float32,  # FP32 for CPU
                safety_checker=None,
                requires_safety_checker=False
            ).to('cpu')  # Force CPU
            
            self.model_loaded = True
            print("✓ Model loaded on CPU")
            print("  (Inpainting will be slower but won't crash)")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            """Lazy load the Stable Diffusion model"""
            if self.model_loaded:
                return
            
            print("Loading Stable Diffusion inpainting model...")
            print("⏳ This may take a few minutes on first run (downloading ~5GB)...")
            
            try:
                from diffusers import StableDiffusionInpaintPipeline
                
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing()
                
                if self.device == "cuda":
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                    except:
                        pass
                
                self.model_loaded = True
                print("✓ Stable Diffusion model loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load Stable Diffusion: {e}")
                print("Inpainting will be disabled. You can still use detection without it.")
                self.model_loaded = False
    
    def detect_occlusion_regions(self, rgb_image, thermal_image=None, threshold=30):
        """
        Detect occluded regions in RGB image, optionally using thermal data
        
        Args:
            rgb_image: RGB image (numpy array)
            thermal_image: Optional thermal image (numpy array)
            threshold: Darkness threshold for occlusion detection
        
        Returns:
            Binary mask (255=occluded, 0=clear)
        """
        if len(rgb_image.shape) == 3:
            gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_rgb = rgb_image
        
        _, dark_mask = cv2.threshold(gray_rgb, threshold, 255, cv2.THRESH_BINARY_INV)
        
        if thermal_image is not None:
            if thermal_image.shape[:2] != rgb_image.shape[:2]:
                thermal_image = cv2.resize(thermal_image, (rgb_image.shape[1], rgb_image.shape[0]))
            
            if len(thermal_image.shape) == 3:
                thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2GRAY)
            else:
                thermal_gray = thermal_image
            
            _, hot_mask = cv2.threshold(thermal_gray, 100, 255, cv2.THRESH_BINARY)
            occlusion_mask = cv2.bitwise_and(dark_mask, hot_mask)
        else:
            occlusion_mask = dark_mask
        
        kernel = np.ones((7, 7), np.uint8)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=1)
        
        return occlusion_mask
    
    def create_smart_mask(self, image, detection_boxes=None, method='auto'):
        """
        Create intelligent mask for inpainting based on image analysis
        
        Args:
            image: Input image
            detection_boxes: Optional list of detection boxes to focus on
            method: 'auto', 'dark_regions', 'blur_regions'
        
        Returns:
            Binary mask for inpainting
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if method == 'auto' or method == 'dark_regions':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
            
            if detection_boxes:
                for box in detection_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, x1 - 20)
                    y1 = max(0, y1 - 20)
                    x2 = min(w, x2 + 20)
                    y2 = min(h, y2 + 20)
                    
                    roi_mask = dark_mask[y1:y2, x1:x2]
                    if roi_mask.sum() > 100:
                        mask[y1:y2, x1:x2] = roi_mask
            else:
                mask = dark_mask
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def inpaint_image(self, image, mask, object_type='default', 
                 num_inference_steps=30, guidance_scale=7.5):
        """
        Inpaint occluded regions using Stable Diffusion
        """
        if not self.model_loaded:
            self._load_model()
    
        if not self.model_loaded:
            print("⚠️  Inpainting unavailable, returning original image")
            return image
    
        try:
            from PIL import Image as PILImage
            
            print(f"  [INPAINT] Input image: {image.shape}, dtype: {image.dtype}")
            print(f"  [INPAINT] Mask: {mask.shape}, dtype: {mask.dtype}")
            print(f"  [INPAINT] Mask pixels: {mask.sum() / 255:.0f}")
            
            # Convert numpy to PIL
            image_pil = PILImage.fromarray(image.astype(np.uint8))
            mask_pil = PILImage.fromarray(mask.astype(np.uint8))
            
            print(f"  [INPAINT] PIL image size: {image_pil.size}")
            print(f"  [INPAINT] PIL mask size: {mask_pil.size}")
            
            # Resize to 512x512 for Stable Diffusion
            original_size = image_pil.size
            image_pil_resized = image_pil.resize((512, 512), PILImage.LANCZOS)
            mask_pil_resized = mask_pil.resize((512, 512), PILImage.NEAREST)  # Use NEAREST for masks
            
            print(f"  [INPAINT] Resized to: {image_pil_resized.size}")
            
            # Get appropriate prompt
            prompt = self.flir_prompts.get(object_type, self.flir_prompts['default'])
            negative_prompt = "blurry, low quality, distorted, unrealistic, dark, occluded, artifacts"
            
            print(f"  [INPAINT] Prompt: '{prompt[:60]}...'")
            print(f"  [INPAINT] Running Stable Diffusion ({num_inference_steps} steps)...")
            
            # Run inpainting
            result = self.pipe(
                prompt=prompt,
                image=image_pil_resized,
                mask_image=mask_pil_resized,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                strength=0.99  # ADD THIS: How much to change (0.99 = maximum change)
            ).images[0]
            
            print(f"  [INPAINT] Stable Diffusion complete")
            print(f"  [INPAINT] Result size: {result.size}")
            
            # Resize back to original
            result = result.resize(original_size, PILImage.LANCZOS)
            
            # Convert back to numpy
            result_np = np.array(result)
            
            print(f"  [INPAINT] Final result: {result_np.shape}, dtype: {result_np.dtype}")
            
            # Check if result is different from input
            diff = np.abs(image.astype(float) - result_np.astype(float)).mean()
            print(f"  [INPAINT] Pixel difference from original: {diff:.2f}")
            
            if diff < 1.0:
                print("  ⚠️  WARNING: Very small change! Inpainting may not have worked.")
            
            return result_np
            
        except Exception as e:
            print(f"❌ Inpainting error: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def reconstruct_target(self, rgb_image, thermal_image=None, 
                          detection_boxes=None, object_type='default'):
        """
        Complete reconstruction pipeline for FLIR ADAS
        
        Args:
            rgb_image: RGB image (numpy array)
            thermal_image: Optional thermal image
            detection_boxes: Optional detection boxes to focus reconstruction
            object_type: Type of object ('person', 'car', etc.)
        
        Returns:
            (reconstructed_image, occlusion_mask)
        """
        print("  Analyzing image for occlusions...")
        
        if thermal_image is not None:
            mask = self.detect_occlusion_regions(rgb_image, thermal_image)
        else:
            mask = self.create_smart_mask(rgb_image, detection_boxes)
        
        occlusion_area = mask.sum() / 255
        total_area = mask.shape[0] * mask.shape[1]
        occlusion_ratio = occlusion_area / total_area
        
        print(f"  Occlusion detected: {occlusion_ratio*100:.1f}% of image")
        
        if occlusion_ratio < 0.01:
            print("  ✓ No significant occlusion, skipping inpainting")
            return rgb_image, mask
        
        if occlusion_ratio > 0.5:
            print("  ⚠️  Heavy occlusion detected, inpainting may be limited")
        
        print("  🎨 Reconstructing occluded regions...")
        reconstructed = self.inpaint_image(rgb_image, mask, object_type=object_type, num_inference_steps=30)
        
        print("  ✓ Reconstruction complete")
        
        return reconstructed, mask


if __name__ == "__main__":
    print("Inpainting module - Use with complete_pipeline.py")