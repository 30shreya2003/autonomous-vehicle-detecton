"""
Debug inpainting issues
"""

import cv2
import numpy as np
from pathlib import Path
from inpainting_module import TargetInpainter
import matplotlib.pyplot as plt

def debug_inpainting():
    print("="*70)
    print("  INPAINTING DEBUG TEST")
    print("="*70)
    
    # 1. Initialize
    print("\n[1/6] Initializing inpainter...")
    inpainter = TargetInpainter(device='cuda')
    print(f"  Device: {inpainter.device}")
    print(f"  Model loaded initially: {inpainter.model_loaded}")
    
    # 2. Create simple test image
    print("\n[2/6] Creating test image...")
    # Create a simple image (blue sky, green ground)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[:256, :, :] = [135, 206, 235]  # Sky blue
    image[256:, :, :] = [34, 139, 34]    # Forest green
    
    # Add a red square in center
    image[200:312, 200:312] = [255, 0, 0]  # Red square
    
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: {image.min()} to {image.max()}")
    
    # 3. Create obvious mask
    print("\n[3/6] Creating mask...")
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Mask the red square
    mask[200:312, 200:312] = 255
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask dtype: {mask.dtype}")
    print(f"  Masked pixels: {mask.sum() / 255}")
    print(f"  Mask ratio: {(mask.sum() / 255) / (512 * 512) * 100:.1f}%")
    
    # 4. Check model loading
    print("\n[4/6] Loading Stable Diffusion model...")
    inpainter._load_model()
    print(f"  Model loaded after _load_model(): {inpainter.model_loaded}")
    
    if not inpainter.model_loaded:
        print("\n❌ MODEL NOT LOADED!")
        print("  Inpainting cannot work without the model.")
        print("\n  Possible reasons:")
        print("  1. diffusers not installed: pip install diffusers")
        print("  2. transformers not installed: pip install transformers")
        print("  3. CUDA out of memory")
        print("  4. Model download failed")
        return
    
    # 5. Run inpainting
    print("\n[5/6] Running inpainting...")
    print("  This will take 15-30 seconds...")
    
    try:
        reconstructed = inpainter.inpaint_image(
            image,
            mask,
            object_type='default',
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Reconstructed dtype: {reconstructed.dtype}")
        print(f"  Reconstructed range: {reconstructed.min()} to {reconstructed.max()}")
        
        # 6. Compare results
        print("\n[6/6] Analyzing results...")
        
        # Check if images are identical
        identical = np.array_equal(image, reconstructed)
        print(f"  Images identical: {identical}")
        
        if identical:
            print("\n❌ PROBLEM: Images are identical!")
            print("  Inpainting did not change anything.")
        else:
            # Calculate difference
            diff = np.abs(image.astype(float) - reconstructed.astype(float)).mean()
            print(f"  Average pixel difference: {diff:.2f}")
            
            # Check masked region difference
            masked_region_orig = image[200:312, 200:312]
            masked_region_recon = reconstructed[200:312, 200:312]
            masked_diff = np.abs(masked_region_orig.astype(float) - masked_region_recon.astype(float)).mean()
            print(f"  Masked region difference: {masked_diff:.2f}")
            
            if masked_diff < 1.0:
                print("\n⚠️  WARNING: Very small change in masked region!")
            else:
                print("\n✅ SUCCESS: Inpainting worked!")
        
        # 7. Save results
        output_dir = Path('debug_inpainting_results')
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'test_original.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / 'test_mask.jpg'), mask)
        cv2.imwrite(str(output_dir / 'test_reconstructed.jpg'), cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        axes[2].imshow(reconstructed)
        axes[2].set_title('Reconstructed')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        
        print(f"\n✓ Results saved to: {output_dir}/")
        print(f"  - test_original.jpg")
        print(f"  - test_mask.jpg")
        print(f"  - test_reconstructed.jpg")
        print(f"  - comparison.png")
        
        print("\n" + "="*70)
        print("OPEN comparison.png TO SEE IF INPAINTING WORKED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ INPAINTING FAILED WITH ERROR:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_inpainting()