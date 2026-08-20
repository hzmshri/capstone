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
    page_title="SubconQC | Pre-Inspection Clearance Portal",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS
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
        border-top: 4px solid #1E293B;
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
        font-size: 18px;
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
        background-color: #1E293B;
        border: 1px solid #0F172A;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.4px;
        border-radius: 4px;
        height: 44px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- PDF VERIFICATION SLIP GENERATOR -----------------
def generate_pdf_report(job_meta, findings, verdict, rfi_clearance, orig_path, annot_path):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", 'B', 15)
    pdf.set_text_color(26, 25, 23)
    pdf.cell(0, 8, "SUBCONTRACTOR SELF-INSPECTION PRE-CHECK SLIP", ln=True, align='L')
    pdf.set_font("Helvetica", '', 9)
    pdf.set_text_color(100, 95, 85)
    pdf.cell(0, 5, "Mandatory Pre-RFI Submission Verification | Structural Steel Package", ln=True, align='L')
    pdf.line(10, 24, 200, 24)
    pdf.ln(5)

    pdf.set_font("Helvetica", 'B', 8.5)
    pdf.set_text_color(50, 45, 35)
    pdf.cell(45, 6, f"Subcontractor: {job_meta['subcon_name']}", 1, 0)
    pdf.cell(50, 6, f"Joint / Drawing ID: {job_meta['drawing_ref']}", 1, 0)
    pdf.cell(45, 6, f"Trade Supervisor: {job_meta['supervisor']}", 1, 0)
    pdf.cell(50, 6, f"Welder Stamp: {job_meta['welder_id']}", 1, 1)

    pdf.cell(95, 6, f"Pre-Check Time: {job_meta['timestamp']}", 1, 0)
    pdf.cell(95, 6, f"Project No: {job_meta['job_no']}", 1, 1)
    pdf.ln(4)

    pdf.set_font("Helvetica", 'B', 10)
    pdf.set_text_color(26, 25, 23)
    pdf.cell(0, 6, "1. RFI CALL-OUT CLEARANCE STATUS", ln=True)
    
    if verdict == "PASS":
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(22, 163, 74)
        pdf.cell(190, 8, "STATUS: CLEARED FOR MAIN CON RFI CALL-OUT (No Surface Non-Conformities)", 1, 1, 'C', fill=True)
    else:
        pdf.set_fill_color(254, 242, 242)
        pdf.set_text_color(220, 38, 38)
        pdf.cell(190, 8, "STATUS: HOLD - REWORK REQUIRED (Do Not Call Main Con / RTO)", 1, 1, 'C', fill=True)

    pdf.ln(4)
    pdf.set_text_color(26, 25, 23)
    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "2. FIELD SEAM CAPTURE & VISION OVERLAY", ln=True)
    y_img = pdf.get_y()
    if os.path.exists(orig_path) and os.path.exists(annot_path):
        pdf.image(orig_path, x=10, y=y_img, w=90, h=55)
        pdf.image(annot_path, x=105, y=y_img, w=90, h=55)
        pdf.set_y(y_img + 57)

    pdf.set_font("Helvetica", 'B', 10)
    pdf.cell(0, 6, "3. PRE-INSPECTION DEFECT LOG", ln=True)
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(85, 6, "Issue Description", 1, 0)
    pdf.cell(30, 6, "Est. Dim (mm)", 1, 0)
    pdf.cell(75, 6, "Required Subcon Action", 1, 1)

    pdf.set_font("Helvetica", '', 8)
    if findings:
        for f in findings:
            pdf.cell(85, 6, str(f["Defect"])[:45], 1, 0)
            pdf.cell(30, 6, str(f["Max Dimension (mm)"]), 1, 0)
            pdf.cell(75, 6, "Grind flush & re-weld before RFI", 1, 1)
    else:
        pdf.cell(190, 6, "Visual inspection criteria satisfied. Uniform bead profile.", 1, 1)

    pdf.ln(8)
    pdf.set_font("Helvetica", 'B', 8)
    pdf.cell(95, 12, "Subcon Trade Foreman Sign: ___________________", 0, 0)
    pdf.cell(95, 12, "Main Con Received By: ________________________", 0, 1)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 24px;">📋</span>
        <div>
            <div style="font-size: 16px; font-weight: 700; color: #1E1D1A;">SubconQC Portal</div>
            <div style="font-size: 11px; color: #7A7468; font-family: 'JetBrains Mono';">Pre-RFI Self-Inspection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### **Trade Submission Info**")
    subcon_name = st.text_input("Subcontractor Firm", value="Titan Steel Works Pte Ltd")
    supervisor = st.text_input("Trade Foreman / Supervisor", value="T. Tan (Foreman)")
    job_no = st.text_input("Main Contract Project No.", value="WH-2026-ST01")
    drawing_ref = st.text_input("Joint ID / Grid Line", value="GL-C4-W02")
    welder_id = st.text_input("Welder Stamp ID", value="W-042")

    st.divider()
    st.markdown("### **Calibration & Scale**")
    workmanship_sens = st.slider("Strictness Factor", 0.5, 2.0, 1.0, 0.1)
    mm_per_pixel = st.number_input("Optical Calibration (mm/px)", value=0.0500, format="%.4f", step=0.005)

# ----------------- MAIN VIEW -----------------
detector = WeldDefectDetector()

st.markdown(f"""
<div class="job-header-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="job-title">Subcontractor Pre-Inspection Self-Check Portal</div>
        <span style="background: #E2E8F0; padding: 4px 10px; border-radius: 4px; font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 600; color: #334155;">PRE-RFI SCREENING</span>
    </div>
    <div class="meta-grid">
        <div class="meta-item">SUBCONTRACTOR<b>{subcon_name}</b></div>
        <div class="meta-item">GRID / JOINT ID<b>{drawing_ref}</b></div>
        <div class="meta-item">WELDER STAMP<b>{welder_id}</b></div>
        <div class="meta-item">TRADE FOREMAN<b>{supervisor}</b></div>
        <div class="meta-item">CAPTURE TIME<b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Capture & Upload Completed Weld Joint Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    img_h, img_w, _ = raw_np.shape

    st.markdown("#### 🎯 Step 1: Align Joint Framing")
    st.caption("Frame the completed weld bead to eliminate parent steel plate reflections.")

    col_x, col_y = st.columns(2)
    with col_x:
        x_range = st.slider("Horizontal Area (% Width)", 0, 100, (15, 85))
    with col_y:
        y_range = st.slider("Vertical Area (% Height)", 0, 100, (15, 85))

    rx = int((x_range[0] / 100.0) * img_w)
    ry = int((y_range[0] / 100.0) * img_h)
    rw = int(((x_range[1] - x_range[0]) / 100.0) * img_w)
    rh = int(((y_range[1] - y_range[0]) / 100.0) * img_h)
    roi_box = (rx, ry, max(rw, 10), max(rh, 10))

    preview_np = raw_np.copy()
    cv2.rectangle(preview_np, (rx, ry), (rx + rw, ry + rh), (30, 41, 59), 3)
    cv2.putText(preview_np, "SUBCON TARGET ROI", (rx, max(ry - 12, 25)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 41, 59), 2)

    st.image(preview_np, caption="Subcon Field Capture Framing", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN PRE-RFI COMPLIANCE CHECK", type="primary", use_container_width=True):
        with st.spinner("Checking bead geometry, cold lap risk, and surface voids..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        st.markdown("<br>", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            color = "#16A34A" if overall_verdict == "PASS" else "#DC2626"
            label = "CLEARED FOR RFI" if overall_verdict == "PASS" else "HOLD - REWORK"
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">Call-Out Clearance</div>
                <div class="metric-val-txt" style="color: {color};">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">Defects to Rectify</div>
                <div class="metric-val-txt" style="color: #1A1917;">{len(findings)}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            ready_pct = "100%" if overall_verdict == "PASS" else f"{max(100 - len(findings) * 40, 30)}%"
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label-txt">RFI Readiness</div>
                <div class="metric-val-txt" style="color: {color};">{ready_pct}</div>
            </div>
            """, unsafe_allow_html=True)

        if overall_verdict == "PASS":
            st.markdown("""
            <div class="verdict-pass">
                ✓ PRE-CHECK CLEARED: Weld profile is uniform and within tolerance. Proceed to submit formal RFI to Main Contractor.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict-fail">
                ✕ PRE-CHECK FAILED: Workmanship defects detected. Grind out and touch up before requesting Main Con / RTO inspection.
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Subcon Field Capture")
            st.image(preview_np, use_container_width=True)

        with col2:
            st.markdown("##### Pre-Check Vision Overlay")
            st.image(annotated_np, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📑 Pre-Inspection Verification Export")

        temp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_annot = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_orig.name, cv2.cvtColor(preview_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_annot.name, cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR))

        job_meta = {
            "job_no": job_no,
            "drawing_ref": drawing_ref,
            "welder_id": welder_id,
            "subcon_name": subcon_name,
            "supervisor": supervisor,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        pdf_path = generate_pdf_report(
            job_meta, findings, overall_verdict, "RFI Clearance", temp_orig.name, temp_annot.name
        )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📄 DOWNLOAD SUBCON PRE-CHECK SLIP (PDF ATTACHMENT FOR RFI)",
            data=pdf_bytes,
            file_name=f"Subcon_PreCheck_{drawing_ref}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary"
        )
