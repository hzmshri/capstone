import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
from weld_analyzer import WeldDefectDetector

# Page Configuration
st.set_page_config(
    page_title="WeldQC System | Structural QA/QC",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Bronze & Brushed Titanium Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Canvas Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #63615A;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(200, 185, 160, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(0, 0, 0, 0.18) 0%, transparent 50%),
            repeating-linear-gradient(45deg, rgba(255,255,255,0.015) 0, rgba(255,255,255,0.015) 1px, transparent 0, transparent 40px);
        color: #F1EFEA;
    }

    /* Left Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #171615 !important;
        border-right: 1px solid #2B2824;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #2D2A26;
        margin: 1.2rem 0;
    }

    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #C8BAA8;
    }

    /* Top Main Title & Subtitle */
    .main-title {
        color: #F4F1EA;
        font-size: 2.2rem;
        font-weight: 500;
        letter-spacing: -0.5px;
        margin-bottom: 0.35rem;
    }
    
    .meta-subtitle {
        color: #D3C9BC;
        font-size: 0.88rem;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }
    
    .meta-subtitle b {
        color: #EDE6DC;
    }

    /* Bronze Sliders & Accent Elements */
    .stSlider > div > div > div > div {
        background-color: #B87A4F !important;
    }
    
    div[data-testid="stThumbValue"] {
        color: #DFC5AA !important;
    }

    /* Brushed Metal Upload Area */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #242528 0%, #1A1A1C 100%);
        border: 1px solid #383735;
        border-radius: 8px;
        padding: 6px 14px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 14px rgba(0,0,0,0.35);
    }
    
    div[data-testid="stFileUploader"] section {
        padding: 0px !important;
        background: transparent !important;
    }
    
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(180deg, #6E4933 0%, #4D3322 100%) !important;
        border: 1px solid #8C5F43 !important;
        color: #F7E9DC !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.35rem 1.2rem !important;
    }

    /* Status Result Banners */
    .pass-card {
        background: rgba(46, 125, 50, 0.22);
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 16px 20px;
        color: #E8F5E9;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.2rem 0;
    }
    
    .fail-card {
        background: rgba(198, 40, 40, 0.22);
        border: 1px solid #EF5350;
        border-radius: 8px;
        padding: 16px 20px;
        color: #FFEBEE;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.2rem 0;
    }

    /* Action Primary Button */
    div.stButton > button:first-child {
        background: linear-gradient(180deg, #8E5632 0%, #5E371E 100%);
        border: 1px solid #AB6B40;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 6px;
        height: 44px;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(180deg, #A2643B 0%, #6E4124 100%);
        border-color: #C78253;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # Custom Brand Flame/Circuit Logo
    st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C10.5 5 8 7.5 8 11C8 13.5 9.5 15.5 12 16C10.5 14 11 12 12.5 10C13.5 12 15 13 15 15C15 17.5 13 22 7 20C12 23 18 20 18 14C18 9 14.5 5 12 2Z" fill="url(#copper_grad)"/>
            <circle cx="12" cy="14" r="1.5" fill="#E8A87C"/>
            <path d="M12 15.5V18M9.5 17L12 15.5M14.5 17L12 15.5" stroke="#E8A87C" stroke-width="1.2" stroke-linecap="round"/>
            <defs>
                <linearGradient id="copper_grad" x1="8" y1="2" x2="18" y2="22" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#E29566"/>
                    <stop offset="0.5" stop-color="#B87346"/>
                    <stop offset="1" stop-color="#733B19"/>
                </linearGradient>
            </defs>
        </svg>
        <div style="color: #D69768; font-size: 1.25rem; font-weight: 600; letter-spacing: 0.5px; margin-top: 6px;">
            WeldQC System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Inspection Settings")
    workmanship_sens = st.slider("Defect Sensitivity", 0.5, 2.0, 1.0, 0.1, 
                                help="Adjust threshold sensitivity for bead width variance and lumpiness.")
    mm_per_pixel = st.number_input("Optical Calibration (mm/px)", value=0.0500, format="%.4f", step=0.005)

    st.markdown("---")
    st.markdown("### Project Reference")
    inspector_name = st.text_input("Inspector ID / Name", value="ENG-QC-01")
    joint_ref = st.text_input("Joint ID / Tag", value="JT-WELD-2026")
    standard_code = st.selectbox("Standard Specification", ["AWS D1.1 (Structural Steel)", "ISO 5817 (Level B)", "ISO 5817 (Level C)"])

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">Weld Defect Detection & Structural QA/QC System</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="meta-subtitle">
    Standard: <b>{standard_code}</b> &nbsp;|&nbsp; 
    Reference: <b>{joint_ref}</b> &nbsp;|&nbsp; 
    Inspector: <b>{inspector_name}</b> &nbsp;|&nbsp; 
    Date: <b>{datetime.now().strftime('%d %b %Y')}</b>
</div>
""", unsafe_allow_html=True)

detector = WeldDefectDetector()

st.markdown("<p style='color: #ECE7DF; font-size: 0.95rem; margin-bottom: 6px;'>Upload Steel Weld Capture (JPG, PNG, BMP)</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    img_h, img_w, _ = raw_np.shape

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 1. Target Weld Seam Framing")
    st.caption("Adjust sliders to box the active bead and isolate background parent metal reflections.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x_range = st.slider("Horizontal Seam Framing (% Width)", 0, 100, (15, 85))
    with col_y:
        y_range = st.slider("Vertical Seam Framing (% Height)", 0, 100, (15, 85))

    rx = int((x_range[0] / 100.0) * img_w)
    ry = int((y_range[0] / 100.0) * img_h)
    rw = int(((x_range[1] - x_range[0]) / 100.0) * img_w)
    rh = int(((y_range[1] - y_range[0]) / 100.0) * img_h)
    roi_box = (rx, ry, max(rw, 10), max(rh, 10))

    preview_np = raw_np.copy()
    cv2.rectangle(preview_np, (rx, ry), (rx + rw, ry + rh), (218, 90, 45), 3)
    cv2.putText(preview_np, "TARGET SEAM ROI", (rx, max(ry - 12, 25)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (218, 90, 45), 2)

    st.image(preview_np, caption="Target ROI Alignment", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run QA/QC Joint Analysis", type="primary", use_container_width=True):
        with st.spinner("Executing structural geometry & defect inspection..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        # Result Verdict Banner
        if overall_verdict == "PASS":
            st.markdown("""
            <div class="pass-card">
                ✓ VERDICT: PASS — Weld Bead Profile Conforms to AWS D1.1 Acceptance Criteria
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fail-card">
                ✕ VERDICT: FAIL — Non-Conformities Detected (Rework / Rectification Required)
            </div>
            """, unsafe_allow_html=True)

        # Inspection Visuals
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Target Joint ROI")
            st.image(preview_np, use_container_width=True)

        with col2:
            st.markdown("##### Annotated QA/QC Evaluation")
            st.image(annotated_np, use_container_width=True)

        # Log Table
        st.markdown("#### 📋 Non-Conformity & Workmanship Log")
        if findings:
            df = pd.DataFrame(findings)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No structural defects or severe workmanship issues detected within the selected seam.")
