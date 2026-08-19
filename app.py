import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
from weld_analyzer import WeldDefectDetector

# Page Setup
st.set_page_config(
    page_title="Welding Detector | Structural Field Inspection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Architectural Cream & Steel Structure Layout CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Body & Steel Truss Blueprint Background */
    .stApp {
        background-color: #F6F4EE;
        background-image: 
            /* Steel Structural Truss Grid Vector Pattern */
            radial-gradient(#D6D1C4 0.75px, transparent 0.75px),
            linear-gradient(to right, rgba(190, 182, 168, 0.15) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(190, 182, 168, 0.15) 1px, transparent 1px);
        background-size: 24px 24px, 120px 120px, 120px 120px;
        color: #24221F;
        font-family: 'Space Grotesk', -apple-system, sans-serif;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ECE8DE !important;
        border-right: 1.5px solid #D8D2C4;
        box-shadow: 2px 0 12px rgba(0,0,0,0.03);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #38342E;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Enterprise Job Header */
    .job-header-card {
        background: #FFFFFF;
        border: 1px solid #DCD6C8;
        border-top: 4px solid #C44536;
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px rgba(50, 45, 35, 0.04);
    }
    
    .job-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #1A1917;
        margin: 0;
    }

    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid #EEE9DF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: #635E54;
    }

    .meta-item b {
        color: #1A1917;
        display: block;
        font-size: 12.5px;
        margin-top: 2px;
    }

    /* Structural KPI Metric Badges */
    .metric-panel {
        background: #141414;
        border: 1px solid #DCD6C8;
        border-radius: 6px;
        padding: 14px 18px;
        text-align: left;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    
    .metric-label-txt {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #7A7468;
    }
    
    .metric-val-txt {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Pass / Fail Compliance Cards */
    .verdict-pass {
        background: #F0FDF4;
        border: 1.5px solid #22C55E;
        border-left: 6px solid #16A34A;
        border-radius: 6px;
        padding: 16px 20px;
        color: #15803D;
        font-weight: 700;
        font-size: 16px;
        margin: 16px 0;
    }

    .verdict-fail {
        background: #FEF2F2;
        border: 1.5px solid #EF4444;
        border-left: 6px solid #DC2626;
        border-radius: 6px;
        padding: 16px 20px;
        color: #B91C1C;
        font-weight: 700;
        font-size: 16px;
        margin: 16px 0;
    }

    /* Industrial Inputs & Sliders */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #D2CBBD !important;
        border-radius: 4px !important;
        color: #1E1D1A !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }

    /* Primary Action CTA Button */
    div.stButton > button:first-child {
        background-color: #C44536;
        border: 1px solid #A8382B;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.4px;
        border-radius: 4px;
        height: 44px;
        box-shadow: 0 2px 6px rgba(196, 69, 54, 0.25);
    }
    
    div.stButton > button:first-child:hover {
        background-color: #AD3B2D;
        border-color: #8E2E23;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR: SYSTEM & SENSOR CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 24px;">📐</span>
        <div>
            <div style="font-size: 16px; font-weight: 700; color: #1E1D1A;">Welding Detector</div>
            <div style="font-size: 11px; color: #7A7468; font-family: 'JetBrains Mono';">v3.4.1 | AWS D1.1-2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### **Job Metadata**")
    job_no = st.text_input("Project / Job No.", value="PRJ-STEEL-7702")
    drawing_ref = st.text_input("Drawing / Joint ID", value="DWG-ST-04-W2")
    welder_id = st.text_input("Welder Stamp ID", value="WLD-8819")
    inspector_id = st.text_input("QC Lead Inspector", value="ENG-HAZZIQ")
    standard_spec = st.selectbox(
        "Design Standard Code", 
        ["AWS D1.1 (Structural Welding Code - Steel)", "EN ISO 5817 (Level B - Stringent)", "EN ISO 5817 (Level C - Moderate)"]
    )
    
    st.divider()
    
    st.markdown("### **Sensor & Optical Calibration**")
    workmanship_sens = st.slider("Tolerance Strictness", 0.5, 2.0, 1.0, 0.1, 
                                help="Adjust standard deviation sensitivity for weld toe and bead roughness.")
    mm_per_pixel = st.number_input("Optical Scale Calibration (mm/px)", value=0.0500, format="%.4f", step=0.005)

# ----------------- MAIN INTERFACE -----------------
detector = WeldDefectDetector()

# Official Structural Header Card
st.markdown(f"""
<div class="job-header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="job-title">Structural Joint NDT / Visual Inspection Report</div>
        <span style="background: #EEE9DF; padding: 4px 10px; border-radius: 4px; font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 600; color: #4A463E;">STATUS: ACTIVE QC SESSION</span>
    </div>
    <div class="meta-grid">
        <div class="meta-item">PROJECT REF<b>{job_no}</b></div>
        <div class="meta-item">JOINT / DRAWING<b>{drawing_ref}</b></div>
        <div class="meta-item">WELDER STAMP<b>{welder_id}</b></div>
        <div class="meta-item">QC INSPECTOR<b>{inspector_id}</b></div>
        <div class="meta-item">TIMESTAMP<b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Field Capture Uploader
uploaded_file = st.file_uploader("Upload Steel Weld Joint Capture (High-Res JPG, PNG, BMP)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    img_h, img_w, _ = raw_np.shape

    st.markdown("#### 🎯 Step 1: Frame Weld Seam Region of Interest (ROI)")
    st.caption("Adjust the sliders below to isolate the active weld bead from parent steel plates and background fixtures.")

    col_x, col_y = st.columns(2)
    with col_x:
        x_range = st.slider("Horizontal Seam Area (% of Width)", 0, 100, (15, 85))
    with col_y:
        y_range = st.slider("Vertical Seam Area (% of Height)", 0, 100, (15, 85))

    rx = int((x_range[0] / 100.0) * img_w)
    ry = int((y_range[0] / 100.0) * img_h)
    rw = int(((x_range[1] - x_range[0]) / 100.0) * img_w)
    rh = int(((y_range[1] - y_range[0]) / 100.0) * img_h)
    roi_box = (rx, ry, max(rw, 10), max(rh, 10))

    preview_np = raw_np.copy()
    cv2.rectangle(preview_np, (rx, ry), (rx + rw, ry + rh), (196, 69, 54), 3)
    cv2.putText(preview_np, "FIELD TARGET ROI", (rx, max(ry - 12, 25)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (196, 69, 54), 2)

    st.image(preview_np, caption="Field Capture ROI Alignment Overlay", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2: Trigger Evaluation
    if st.button("⚡ EXECUTE STRUCTURAL INTEGRITY ANALYSIS", type="primary", use_container_width=True):
        with st.spinner("Processing optical scale, bead morphology, and defect tolerances..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Enterprise KPI Dashboard
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            color = "#16A34A" if overall_verdict == "PASS" else "#DC2626"
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">Joint Acceptance</div>
                <div class="metric-val-txt" style="color: {color};">{overall_verdict}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">Defect Flags</div>
                <div class="metric-val-txt" style="color: #1A1917;">{len(findings)}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            score = 98 if overall_verdict == "PASS" else max(100 - len(findings) * 35, 45)
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">QA Integrity Score</div>
                <div class="metric-val-txt" style="color: {color};">{score}/100</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">Governing Code</div>
                <div class="metric-val-txt" style="color: #635E54; font-size: 15px; padding-top: 5px;">AWS D1.1:2025</div>
            </div>
            """, unsafe_allow_html=True)

        # Formatted Verdict Card
        if overall_verdict == "PASS":
            st.markdown("""
            <div class="verdict-pass">
                ✓ ACCEPTANCE VERDICT: PASS — Weldment conforms with AWS D1.1 visual inspection tolerances. Ready for sign-off.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict-fail">
                ✕ ACCEPTANCE VERDICT: FAIL — Non-conformities exceed allowable structural tolerance. Rework / gouging required.
            </div>
            """, unsafe_allow_html=True)

        # Visual Side-by-Side Verification
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Raw Alignment Capture")
            st.image(preview_np, use_container_width=True)

        with col2:
            st.markdown("##### NDT Computer Vision Overlay")
            st.image(annotated_np, use_container_width=True)

        # Findings Log Table
        st.markdown("#### 📋 Non-Conformance & Workmanship Record")
        if findings:
            df = pd.DataFrame(findings)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No surface discontinuities, porosity clusters, or profile irregularities detected in selected seam.")
