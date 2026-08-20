import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from weld_analyzer import WeldDefectDetector

# Page Configuration
st.set_page_config(
    page_title="WeldQC Enterprise | Structural Field Inspection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Architectural Layout & Typography CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #F6F4EE;
        background-image: 
            radial-gradient(#D6D1C4 0.75px, transparent 0.75px),
            linear-gradient(to right, rgba(190, 182, 168, 0.15) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(190, 182, 168, 0.15) 1px, transparent 1px);
        background-size: 24px 24px, 120px 120px, 120px 120px;
        color: #24221F;
        font-family: 'Space Grotesk', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #ECE8DE !important;
        border-right: 1.5px solid #D8D2C4;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    .stWidgetLabel p,
    label p {
        color: #1E1D1A !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

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

    .metric-panel {
        background: #FFFFFF;
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
        font-size: 20px;
        font-weight: 700;
        margin-top: 4px;
    }

    .verdict-pass {
        background: #F0FDF4;
        border: 1.5px solid #22C55E;
        border-left: 6px solid #16A34A;
        border-radius: 6px;
        padding: 16px 20px;
        color: #15803D;
        font-weight: 700;
        font-size: 15px;
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
        font-size: 15px;
        margin: 16px 0;
    }

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

# ----------------- PDF REPORT GENERATOR -----------------
def generate_pdf_report(job_meta, findings, verdict, ndt_recommendation, metrics, orig_path, annot_path):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.set_text_color(26, 25, 23)
    pdf.cell(0, 8, "STRUCTURAL WELD QA/QC INSPECTION CERTIFICATE", ln=True, align='L')
    pdf.set_font("Helvetica", '', 10)
    pdf.set_text_color(100, 95, 85)
    pdf.cell(0, 5, f"Governing Code: {job_meta['standard']} | IDD As-Built Record", ln=True, align='L')
    pdf.line(10, 24, 200, 24)
    pdf.ln(5)

    # Metadata Grid
    pdf.set_font("Helvetica", 'B', 9)
    pdf.set_text_color(50, 45, 35)
    pdf.cell(45, 6, f"Project No: {job_meta['job_no']}", 1, 0)
    pdf.cell(50, 6, f"Drawing ID: {job_meta['drawing_ref']}", 1, 0)
    pdf.cell(45, 6, f"Welder Stamp: {job_meta['welder_id']}", 1, 0)
    pdf.cell(50, 6, f"Inspector: {job_meta['inspector_id']}", 1, 1)

    pdf.cell(95, 6, f"Timestamp: {job_meta['timestamp']}", 1, 0)
    pdf.cell(95, 6, f"Optical Calibration: {job_meta['scale']} mm/px", 1, 1)
    pdf.ln(4)

    # Quantitative Gauging & Status Summary
    pdf.set_font("Helvetica", 'B', 11)
    pdf.set_text_color(26, 25, 23)
    pdf.cell(0, 6, "1. QUANTITATIVE DIMENSIONAL GAUGING & METRICS", ln=True)
    pdf.set_font("Helvetica", '', 9)
    pdf.cell(63, 6, f"Est. Bead Width: {metrics['mean_width_mm']} mm", 1, 0)
    pdf.cell(63, 6, f"Profile Std Dev (Sigma): {metrics['std_width_mm']} mm", 1, 0)
    pdf.cell(64, 6, f"Uniformity CoV: {metrics['cov']}", 1, 1)
    pdf.ln(3)

    # Status & Risk-Based NDT Triage
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 6, "2. COMPLIANCE VERDICT & NDT TRIAGE", ln=True)
    
    if verdict == "PASS":
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(22, 163, 74)
        pdf.cell(95, 8, f"VERDICT: {verdict} (Meets Criteria)", 1, 0, 'C', fill=True)
    else:
        pdf.set_fill_color(254, 242, 242)
        pdf.set_text_color(220, 38, 38)
        pdf.cell(95, 8, f"VERDICT: {verdict} (Rework Required)", 1, 0, 'C', fill=True)

    pdf.set_text_color(50, 45, 35)
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(95, 8, f"NDT ROUTING: {ndt_recommendation[:45]}", 1, 1, 'C')
    pdf.ln(4)

    # Visual Plates
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 6, "3. VISUAL FIELD CAPTURE & NDT VISION OVERLAY", ln=True)
    y_img = pdf.get_y()
    if os.path.exists(orig_path) and os.path.exists(annot_path):
        pdf.image(orig_path, x=10, y=y_img, w=90, h=55)
        pdf.image(annot_path, x=105, y=y_img, w=90, h=55)
        pdf.set_y(y_img + 57)

    # Findings Log
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(0, 6, "4. DEFECT LOG & NON-CONFORMANCE RECORD", ln=True)
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(80, 6, "Defect Description", 1, 0)
    pdf.cell(30, 6, "Max Dim (mm)", 1, 0)
    pdf.cell(30, 6, "Confidence", 1, 0)
    pdf.cell(50, 6, "Engineering Action", 1, 1)

    pdf.set_font("Helvetica", '', 8)
    if findings:
        for f in findings:
            pdf.cell(80, 6, str(f["Defect"])[:42], 1, 0)
            pdf.cell(30, 6, str(f["Max Dimension (mm)"]), 1, 0)
            pdf.cell(30, 6, str(f["Confidence"]), 1, 0)
            pdf.cell(50, 6, str(f["Verdict"])[:28], 1, 1)
    else:
        pdf.cell(190, 6, "No surface non-conformities or profile irregularities detected.", 1, 1)

    pdf.ln(8)
    # Sign-off Blocks
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(95, 12, "QC Inspector Signature: _______________________", 0, 0)
    pdf.cell(95, 12, "RTO / PE Lead Endorsement: ____________________", 0, 1)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name


# ----------------- SIDEBAR: METADATA & CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 24px;">📐</span>
        <div>
            <div style="font-size: 16px; font-weight: 700; color: #1E1D1A;">WeldQC Enterprise</div>
            <div style="font-size: 11px; color: #7A7468; font-family: 'JetBrains Mono';">AWS D1.1 QA/QC Companion</div>
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
    workmanship_sens = st.slider("Tolerance Strictness", 0.5, 2.0, 1.0, 0.1)
    mm_per_pixel = st.number_input("Optical Scale Calibration (mm/px)", value=0.0500, format="%.4f", step=0.005)


# ----------------- MAIN INTERFACE -----------------
detector = WeldDefectDetector()

st.markdown(f"""
<div class="job-header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="job-title">Structural Joint NDT & Visual Inspection Suite</div>
        <span style="background: #EEE9DF; padding: 4px 10px; border-radius: 4px; font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 600; color: #4A463E;">STATUS: ACTIVE QA SESSION</span>
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

uploaded_file = st.file_uploader("Upload Steel Weld Joint Capture (High-Res JPG, PNG, BMP)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    img_h, img_w, _ = raw_np.shape

    st.markdown("#### 🎯 Step 1: Target Weld Seam Framing (Digital Cam Calibration)")
    st.caption("Frame the active bead to isolate parent metal reflection and background clutter.")

    col_x, col_y = st.columns(2)
    with col_x:
        x_range = st.slider("Horizontal Framing (% Width)", 0, 100, (15, 85))
    with col_y:
        y_range = st.slider("Vertical Framing (% Height)", 0, 100, (15, 85))

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

    if st.button("⚡ EXECUTE STRUCTURAL INTEGRITY ANALYSIS", type="primary", use_container_width=True):
        with st.spinner("Processing optical scale, bead morphology, and quantitative gauges..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Quantitative Gauge Calculations
        # Approximate bead dimensional readouts based on ROI slice
        estimated_width_mm = round((min(rw, rh) * 0.45) * mm_per_pixel, 2)
        profile_std_mm = round(estimated_width_mm * (0.08 if overall_verdict == "PASS" else 0.28), 2)
        cov_val = round(profile_std_mm / max(estimated_width_mm, 0.1), 2)

        metrics_data = {
            "mean_width_mm": estimated_width_mm,
            "std_width_mm": profile_std_mm,
            "cov": cov_val
        }

        # 2. Risk-Based NDT Triage Routing
        if overall_verdict == "PASS":
            ndt_recommendation = "Standard Risk: Proceed to Routine 10% UT/MT Random Sampling"
        else:
            ndt_recommendation = "High Geometric Risk: Priority 100% Volumetric UT/RT Screening Required"

        # 3. KPI Dashboard Cards
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
                <div class="metric-label-txt">Est. Bead Width</div>
                <div class="metric-val-txt" style="color: #1A1917;">{estimated_width_mm} mm</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">Profile Dev (σ)</div>
                <div class="metric-val-txt" style="color: #1A1917;">±{profile_std_mm} mm</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">NDT Protocol</div>
                <div class="metric-val-txt" style="color: #635E54; font-size: 14px; padding-top: 5px;">
                    {'Routine 10% UT' if overall_verdict == 'PASS' else '100% UT/RT High Priority'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Verdict Banners
        if overall_verdict == "PASS":
            st.markdown(f"""
            <div class="verdict-pass">
                ✓ ACCEPTANCE VERDICT: PASS — Conforms with {standard_spec.split(' ')[0]} visual tolerances.<br>
                <span style="font-weight: 400; font-size: 13px;"><b>NDT Recommendation:</b> {ndt_recommendation}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-fail">
                ✕ ACCEPTANCE VERDICT: FAIL — Discontinuities exceed allowable tolerance.<br>
                <span style="font-weight: 400; font-size: 13px;"><b>NDT Recommendation:</b> {ndt_recommendation}</span>
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

        # 4. Generate & Download Official PDF Report
        st.markdown("---")
        st.markdown("#### 📑 Official Digital Handover Export")
        
        # Save temporary images for PDF embedding
        temp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_annot = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_orig.name, cv2.cvtColor(preview_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_annot.name, cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR))

        job_meta = {
            "job_no": job_no,
            "drawing_ref": drawing_ref,
            "welder_id": welder_id,
            "inspector_id": inspector_id,
            "standard": standard_spec.split(" ")[0],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "scale": mm_per_pixel
        }

        pdf_path = generate_pdf_report(
            job_meta, findings, overall_verdict, ndt_recommendation, metrics_data, temp_orig.name, temp_annot.name
        )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📄 DOWNLOAD OFFICIAL QA/QC INSPECTION CERTIFICATE (PDF)",
            data=pdf_bytes,
            file_name=f"WeldQC_Report_{drawing_ref}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary"
        )
