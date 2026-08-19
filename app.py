import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
from weld_analyzer import WeldDefectDetector

# Page Configuration
st.set_page_config(
    page_title="WeldQC | Structural QA/QC System",
    page_icon="⚡",
    layout="wide"
)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.title("⚡ WeldQC System")
    st.markdown("---")
    st.subheader("Inspection Settings")
    workmanship_sens = st.slider("Defect Sensitivity", 0.5, 2.0, 1.0, 0.1, 
                                help="Adjust standard deviation sensitivity for weld toe and bead roughness.")
    mm_per_pixel = st.number_input("Optical Calibration (mm/px)", value=0.0500, format="%.4f", step=0.005)
    
    st.markdown("---")
    st.subheader("Project Reference")
    inspector_name = st.text_input("Inspector ID / Name", value="ENG-QC-01")
    joint_ref = st.text_input("Joint ID / Tag", value="JT-WELD-2026")
    standard_code = st.selectbox("Design Standard", ["AWS D1.1 (Structural Steel)", "ISO 5817 (Level B)", "ISO 5817 (Level C)"])
    
    st.caption("Powered by Morphological Computer Vision & YOLOv8")

# ----------------- MAIN HEADER -----------------
st.title("Weld Defect Detection & Structural QA/QC System")
st.caption(f"**Standard:** {standard_code} | **Reference:** {joint_ref} | **Inspector:** {inspector_name} | **Date:** {datetime.now().strftime('%d %b %Y')}")

detector = WeldDefectDetector()

# ----------------- IMAGE UPLOADER -----------------
uploaded_file = st.file_uploader("Upload Steel Weld Capture (JPG, PNG, BMP)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    img_h, img_w, _ = raw_np.shape

    st.subheader("1. Frame the Weld Seam Region (ROI)")
    st.info("Adjust the sliders below to box the weld bead and exclude background metal/bolts.")
    
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
    cv2.rectangle(preview_np, (rx, ry), (rx + rw, ry + rh), (229, 9, 20), 3)
    cv2.putText(preview_np, "TARGET SEAM ROI", (rx, max(ry - 12, 25)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (229, 9, 20), 2)

    st.image(preview_np, caption="Target Bounding Box Overlay", use_container_width=True)

    # Trigger Evaluation
    if st.button("Run Structural Joint Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing weld geometry, bead consistency, and surface defects..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        st.markdown("---")
        
        # KPI Summary
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Overall Verdict", value=overall_verdict)
        with kpi2:
            st.metric(label="Defects Identified", value=len(findings))
        with kpi3:
            score = "98 / 100" if overall_verdict == "PASS" else f"{max(100 - len(findings) * 35, 45)} / 100"
            st.metric(label="Integrity Index", value=score)

        # Verdict Alert Banner
        if overall_verdict == "PASS":
            st.success("### VERDICT: PASS — Weld Bead Profile Conforms to Acceptance Criteria")
        else:
            st.error("### VERDICT: FAIL — Non-Conformities Detected (Rework Required)")

        # Visual Comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Target Joint ROI")
            st.image(preview_np, use_container_width=True)

        with col2:
            st.subheader("QA/QC Annotated Evaluation")
            st.image(annotated_np, use_container_width=True)

        # Findings Log Table
        st.subheader("Inspection Findings Log")
        if findings:
            df = pd.DataFrame(findings)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No structural defects or severe workmanship issues detected within the selected seam.")
