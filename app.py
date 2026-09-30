import os
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
import tempfile

from src.preprocessing import ImagePreprocessor
from src.model import build_model
from src.explainability import ExplainableAI
from src.report import generate_pdf_report

# Streamlit Page Config
st.set_page_config(
    page_title="Early Parkinson's Detection & Visual Explainability",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        color: #1E3A8A;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model(model_path="models/model.pt", device='cpu'):
    model = build_model(num_classes=2, pretrained=False, freeze_backbone=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.success(" Loaded fine-tuned ResNet-18 model.")
    else:
        st.sidebar.warning(" Model checkpoint missing. Operating with ImageNet pretrained weights for demo mode.")
    model.to(device)
    model.eval()
    return model

def main():
    st.markdown('<div class="main-title">Early Parkinson\'s Disease Detection & Grad-CAM Visual Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Point-of-Care Hand Drawing & Tremor Analysis Powered by PyTorch ResNet-18</div>', unsafe_allow_html=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_cached_model("models/model.pt", device=device)
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    explainer = ExplainableAI(model, device=device)

    # Sidebar Options
    st.sidebar.header("⚙️ Configuration & Info")
    st.sidebar.markdown("""
    **Pipeline Workflow:**
    1. **Input:** Spiral / Wave / Handwriting Drawing
    2. **Preprocessing:** Grayscale, Denoising & Adaptive Thresholding
    3. **Model:** ResNet-18 Backbone (Transfer Learning)
    4. **Explainability:** Grad-CAM Heatmap (`layer4[-1]`)
    5. **Output:** Downloadable PDF Diagnostic Report
    """)

    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.subheader("1. Input Image Acquisition")
        uploaded_file = st.file_uploader(
            "Upload spiral or wave stroke drawing (PNG, JPG, JPEG):",
            type=['png', 'jpg', 'jpeg']
        )
        
        # Sample images fallback button
        st.markdown("**Or test with sample images:**")
        sample_cols = st.columns(2)
        sample_choice = None
        if sample_cols[0].button("Sample Healthy Spiral"):
            sample_choice = "dataset/kaggle/drawings/spiral/testing/healthy/010101-healthy.png"
            if not os.path.exists(sample_choice):
                # Search for any healthy sample
                import glob
                healthy_files = glob.glob("dataset/**/healthy*.*", recursive=True)
                sample_choice = healthy_files[0] if healthy_files else None

        if sample_cols[1].button("Sample Parkinson's Spiral"):
            sample_choice = "dataset/kaggle/drawings/spiral/testing/parkinson/010101-parkinson.png"
            if not os.path.exists(sample_choice):
                import glob
                pd_files = glob.glob("dataset/**/parkinson*.*", recursive=True)
                sample_choice = pd_files[0] if pd_files else None

        img_to_process = None
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif sample_choice and os.path.exists(sample_choice):
            img_to_process = cv2.imread(sample_choice)
            st.info(f"Loaded sample image: {os.path.basename(sample_choice)}")

        if img_to_process is not None:
            st.image(cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB), caption="Uploaded Input Drawing", use_container_width=True)

    with col_results:
        st.subheader("2. Inference & Visual Diagnostic Output")
        
        if img_to_process is not None:
            with st.spinner("Executing Preprocessing, ResNet-18 Inference, and Grad-CAM..."):
                # Preprocess
                processed_rgb, input_tensor, stroke_metrics = preprocessor.preprocess_pipeline(img_to_process)
                
                # Model Inference
                tensor_batch = input_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor_batch)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_class_idx = torch.argmax(probs).item()
                    confidence = probs[pred_class_idx].item() * 100.0

                label_str = "Parkinson's Disease" if pred_class_idx == 1 else "Healthy / Control"
                
                # Grad-CAM Heatmap
                heatmap_vis, cam_map = explainer.generate_heatmap(
                    input_tensor=input_tensor,
                    rgb_img=processed_rgb,
                    target_category=pred_class_idx
                )

                # Save temporary images for PDF report
                with tempfile.TemporaryDirectory() as tmp_dir:
                    orig_tmp_path = os.path.join(tmp_dir, "input_preprocessed.png")
                    heatmap_tmp_path = os.path.join(tmp_dir, "gradcam_heatmap.png")
                    pdf_tmp_path = os.path.join(tmp_dir, "parkinson_diagnostic_report.pdf")

                    cv2.imwrite(orig_tmp_path, cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(heatmap_tmp_path, cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR))

                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Prediction", label_str)
                    m2.metric("Confidence", f"{confidence:.2f}%")
                    m3.metric("Stroke Smoothness", f"{stroke_metrics['smoothness_score']}/100")

                    if pred_class_idx == 1:
                        st.error(f"⚠️ Diagnosis Signal Detected: **{label_str}** with **{confidence:.1f}%** confidence.")
                    else:
                        st.success(f" Diagnosis Signal Normal: **{label_str}** with **{confidence:.1f}%** confidence.")

                    # Display Side-by-Side Images
                    st.markdown("### 📊 Side-by-Side Stroke & Heatmap Analysis")
                    vis_col1, vis_col2 = st.columns(2)
                    vis_col1.image(processed_rgb, caption="Preprocessed Stroke Image", use_container_width=True)
                    vis_col2.image(heatmap_vis, caption="Grad-CAM Tremor Activation (layer4)", use_container_width=True)

                    # Generate PDF Report
                    st.markdown("### 📄 Export PDF Diagnostic Report")
                    generate_pdf_report(
                        input_img_path=orig_tmp_path,
                        heatmap_img_path=heatmap_tmp_path,
                        prediction_label=label_str,
                        confidence_score=confidence,
                        stroke_metrics=stroke_metrics,
                        output_pdf_path=pdf_tmp_path
                    )

                    with open(pdf_tmp_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()

                    st.download_button(
                        label="📥 Download Full PDF Diagnostic Report",
                        data=pdf_bytes,
                        file_name=f"Parkinson_Report_{label_str.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
        else:
            st.info(" Please upload a spiral or wave drawing image on the left panel to begin diagnostic inference.")

if __name__ == "__main__":
    main()
