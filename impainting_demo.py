"""
Standalone inpainting demonstration
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from inpainting_module import TargetInpainter

st.set_page_config(page_title="FLIR Inpainting Demo", layout="wide")

st.title("🎨 FLIR ADAS Inpainting Demonstration")

# Initialize inpainter
@st.cache_resource
def load_inpainter():
    return TargetInpainter(device='cuda')

try:
    inpainter = load_inpainter()
    st.sidebar.success("✅ Inpainting model ready")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_np)
    
    # Mask options
    st.sidebar.subheader("Mask Options")
    mask_method = st.sidebar.selectbox(
        "Mask Creation Method",
        ["Auto (Dark Regions)", "Manual Rectangle"]
    )
    
    if mask_method == "Auto (Dark Regions)":
        threshold = st.sidebar.slider("Dark Threshold", 10, 100, 40)
        mask = inpainter.create_smart_mask(image_np, method='dark_regions')
    else:
        # Manual mask
        st.sidebar.write("Draw rectangle on image (coming soon)")
        mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        # Create center rectangle
        h, w = image_np.shape[:2]
        mask[h//4:3*h//4, w//4:3*w//4] = 255
    
    # Show mask
    st.subheader("Occlusion Mask")
    st.image(mask, caption=f"Occluded pixels: {mask.sum() / 255:.0f}")
    
    # Inpainting settings
    st.sidebar.subheader("Inpainting Settings")
    object_type = st.sidebar.selectbox(
        "Object Type",
        ['default', 'person', 'car', 'bike', 'bus', 'truck']
    )
    
    num_steps = st.sidebar.slider("Inference Steps", 10, 50, 30,
                                  help="Higher = better quality, slower")
    
    guidance = st.sidebar.slider("Guidance Scale", 1.0, 15.0, 7.5,
                                 help="How closely to follow prompt")
    
    if st.button("🎨 Run Inpainting", type="primary"):
        with st.spinner("Reconstructing image..."):
            try:
                reconstructed = inpainter.inpaint_image(
                    image_np,
                    mask,
                    object_type=object_type,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance
                )
                
                with col2:
                    st.subheader("Reconstructed Image")
                    st.image(reconstructed)
                
                # Download button
                result_pil = Image.fromarray(reconstructed)
                import io
                buf = io.BytesIO()
                result_pil.save(buf, format='JPEG')
                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name="inpainted.jpg",
                    mime="image/jpeg"
                )
                
                st.success("✓ Inpainting complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("👆 Upload an image to begin")