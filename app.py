"""
Streamlit Web Application for FLIR ADAS Target Recognition
Interactive interface for detection and inpainting
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import time

from detection_model import MultiModalYOLO
from inpainting_module import TargetInpainter

# Page config
st.set_page_config(
    page_title="FLIR ADAS Target Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector(model_path):
    """Load detection model (cached)"""
    detector = MultiModalYOLO(model_size='s')
    detector.load(model_path)
    return detector

@st.cache_resource
def load_inpainter(device):
    """Load inpainting model (cached)"""
    inpainter = TargetInpainter(device=device)
    return inpainter

def main():
    # Header
    st.markdown('<div class="main-header">🎯 FLIR ADAS Target Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Modal Object Detection with Generative Inpainting</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("📁 Model Selection")
    
    # Find available models
    model_paths = []
    if Path('runs/detect/multimodal_yolo/weights/best.pt').exists():
        model_paths.append('runs/detect/multimodal_yolo/weights/best.pt')
    
    results_dir = Path('results')
    if results_dir.exists():
        model_paths.extend(sorted(results_dir.glob('*/best_model.pt')))
    
    if not model_paths:
        st.error("❌ No trained models found! Please train a model first.")
        st.info("Run: `python train_pipeline.py`")
        return
    
    model_path = st.sidebar.selectbox(
        "Select Model",
        model_paths,
        format_func=lambda x: str(x).split('/')[-2] if 'results' in str(x) else "Latest Training"
    )
    
    # Detection settings
    st.sidebar.subheader("🔍 Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = more detections (may include false positives)"
    )
    
    # Inpainting settings
    st.sidebar.subheader("🎨 Inpainting Settings")
    use_inpainting = st.sidebar.checkbox(
        "Enable Inpainting",
        value=False,
        help="Use AI to reconstruct occluded/degraded regions"
    )
    
    if use_inpainting:
        inpaint_threshold = st.sidebar.slider(
            "Inpainting Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Trigger inpainting if confidence below this"
        )
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"🖥️ Using: **{device.upper()}**")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"GPU: {gpu_name}")
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            detector = load_detector(str(model_path))
            st.sidebar.success("✅ Detection model loaded")
            
            if use_inpainting:
                inpainter = load_inpainter(device)
                st.sidebar.success("✅ Inpainting model ready")
            else:
                inpainter = None
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Processing", "ℹ️ About"])
    
    # Tab 1: Single Image
    with tab1:
        st.header("Upload and Process Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image (RGB or Thermal)",
                type=['jpg', 'jpeg', 'png'],
                key="single_image"
            )
            
            if uploaded_file:
                # Load image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Convert to RGB if needed
                if len(image_np.shape) == 2:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                elif image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                
                st.image(image_np, caption="Original Image", use_container_width=True)
                
                if st.button("🚀 Process Image", key="process_single"):
                    with st.spinner("Processing..."):
                        start_time = time.time()
                        
                        # Detection
                        results = detector.detect(image_np, conf_threshold=conf_threshold)
                        detections = results.boxes
                        
                        # Optional inpainting
                        reconstructed = None
                        if use_inpainting and inpainter and len(detections) > 0:
                            avg_conf = detections.conf.mean().item()
                            if avg_conf < inpaint_threshold:
                                st.info("🎨 Applying inpainting (low confidence detected)...")
                                detection_boxes = []
                                for box in detections:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    detection_boxes.append([x1, y1, x2, y2])
                                
                                reconstructed, mask = inpainter.reconstruct_target(
                                    image_np,
                                    detection_boxes=detection_boxes
                                )
                                
                                # Re-detect on reconstructed
                                results = detector.detect(reconstructed, conf_threshold=conf_threshold)
                                detections = results.boxes
                        
                        # Visualize
                        vis_image = (reconstructed if reconstructed is not None else image_np).copy()
                        vis_image = detector.visualize_results(vis_image, results, show_labels=True)
                        
                        process_time = time.time() - start_time
                        
                        # Store in session state
                        st.session_state['processed_image'] = vis_image
                        st.session_state['detections'] = detections
                        st.session_state['process_time'] = process_time
        
        with col2:
            st.subheader("Results")
            
            if 'processed_image' in st.session_state:
                st.image(st.session_state['processed_image'], caption="Processed Image", use_container_width=True)
                
                # Metrics
                detections = st.session_state['detections']
                process_time = st.session_state['process_time']
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("🎯 Detections", len(detections))
                with col_b:
                    st.metric("⏱️ Time", f"{process_time:.2f}s")
                with col_c:
                    fps = 1.0 / process_time if process_time > 0 else 0
                    st.metric("📊 FPS", f"{fps:.1f}")
                
                # Detection details
                if len(detections) > 0:
                    st.subheader("Detection Details")
                    for i, box in enumerate(detections):
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = detector.class_names[cls]
                        
                        st.write(f"**{i+1}.** {class_name} - Confidence: {conf:.2%}")
                
                # Download button
                result_pil = Image.fromarray(st.session_state['processed_image'])
                import io
                buf = io.BytesIO()
                result_pil.save(buf, format='JPEG')
                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )
            else:
                st.info("👆 Upload an image and click 'Process Image' to see results")
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Process Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_images"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process All Images", key="process_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_container = st.container()
                
                total_detections = 0
                total_time = 0
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Load and process
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    
                    if len(image_np.shape) == 2:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                    elif image_np.shape[2] == 4:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                    
                    start_time = time.time()
                    results = detector.detect(image_np, conf_threshold=conf_threshold)
                    vis_image = detector.visualize_results(image_np, results, show_labels=True)
                    process_time = time.time() - start_time
                    
                    total_detections += len(results.boxes)
                    total_time += process_time
                    
                    # Display result
                    with results_container:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image_np, caption=f"Original: {uploaded_file.name}", use_container_width=True)
                        with col2:
                            st.image(vis_image, caption=f"Detections: {len(results.boxes)} ({process_time:.2f}s)", use_container_width=True)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing complete!")
                
                # Summary metrics
                st.success(f"**Summary:** {total_detections} total detections in {total_time:.2f}s (avg: {total_time/len(uploaded_files):.2f}s per image)")
    
    # Tab 3: About
    with tab3:
        st.header("About FLIR ADAS Target Recognition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Features")
            st.markdown("""
            - **Multi-Modal Detection**: RGB and Thermal image support
            - **15 Object Classes**: Person, Car, Bike, Motor, Bus, Truck, and more
            - **Real-time Processing**: GPU-accelerated inference
            - **Generative Inpainting**: AI-powered reconstruction of occluded targets
            - **Batch Processing**: Process multiple images at once
            - **Adjustable Parameters**: Fine-tune confidence thresholds
            """)
            
            st.subheader("📊 Model Information")
            st.markdown(f"""
            - **Architecture**: YOLOv8 (You Only Look Once)
            - **Dataset**: FLIR ADAS (Automotive Detection)
            - **Classes**: {len(detector.class_names)}
            - **Model Path**: `{model_path}`
            """)
        
        with col2:
            st.subheader("🚀 How to Use")
            st.markdown("""
            **Single Image Mode:**
            1. Upload an image (RGB or Thermal)
            2. Adjust confidence threshold if needed
            3. Enable inpainting for degraded images (optional)
            4. Click "Process Image"
            5. Download results
            
            **Batch Processing Mode:**
            1. Upload multiple images
            2. Configure settings
            3. Click "Process All Images"
            4. Review all results
            """)
            
            st.subheader("🔧 Tips")
            st.markdown("""
            - **Low confidence threshold** (0.15-0.25): More detections, may include false positives
            - **High confidence threshold** (0.4-0.6): Fewer but more accurate detections
            - **Enable inpainting**: Use for heavily occluded or low-light images
            - **Batch processing**: Ideal for analyzing datasets
            """)
        
        st.subheader("📚 Class Names")
        cols = st.columns(5)
        for idx, class_name in enumerate(detector.class_names):
            with cols[idx % 5]:
                st.write(f"{idx}. {class_name}")
        
        st.subheader("💻 System Information")
        st.code(f"""
Device: {device.upper()}
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
PyTorch: {torch.__version__}
CUDA Available: {torch.cuda.is_available()}
        """)

if __name__ == "__main__":
    main()