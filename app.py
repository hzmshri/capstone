import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from weld_analyzer import WeldDefectDetector

st.set_page_config(page_title="WeldQC - AI Inspection System", layout="wide")

st.title("Weld Defect Detection & Structural QA/QC System")
st.markdown("Automated vision inspection for steel joint compliance (AWS D1.1 / EN ISO 5817).")

# Sidebar Configuration
st.sidebar.header("Inspection Settings")
workmanship_sens = st.sidebar.slider("Defect Sensitivity (Tolerances)", 0.5, 2.0, 1.0, 0.1)
mm_per_pixel = st.sidebar.number_input("Optical Scale Calibration (mm/px)", value=0.050, format="%.4f")

detector = WeldDefectDetector()

uploaded_file = st.file_uploader("Upload Steel Weld Joint Capture (JPG / PNG / BMP)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)
    img_h, img_w, _ = raw_np.shape

    st.subheader("Step 1: Frame the Weld Seam (Exclude Background & Bolts)")
    
    # Coordinate framing sliders
    col_x, col_y = st.columns(2)
    with col_x:
        x_range = st.slider("Horizontal Seam Area (% of Width)", 0, 100, (15, 85))
    with col_y:
        y_range = st.slider("Vertical Seam Area (% of Height)", 0, 100, (15, 85))

    # Convert percentages to pixel bounding box
    rx = int((x_range[0] / 100.0) * img_w)
    ry = int((y_range[0] / 100.0) * img_h)
    rw = int(((x_range[1] - x_range[0]) / 100.0) * img_w)
    rh = int(((y_range[1] - y_range[0]) / 100.0) * img_h)
    roi_box = (rx, ry, max(rw, 10), max(rh, 10))

    # Live Preview with Bounding Box Overlay
    preview_np = raw_np.copy()
    cv2.rectangle(preview_np, (rx, ry), (rx + rw, ry + rh), (255, 50, 50), 3)
    cv2.putText(preview_np, "Selected Weld Seam ROI", (rx, max(ry - 10, 25)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

    st.image(preview_np, caption="Adjust sliders above to tightly box the weld bead.", use_container_width=True)

    st.divider()

    # Step 2: Run Analysis
    if st.button("Run QA/QC Joint Analysis", type="primary"):
        with st.spinner("Analyzing isolated weld seam geometry..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Analyzed Joint Framing")
            st.image(preview_np, use_container_width=True)

        with col2:
            st.subheader("QA/QC Annotated Verdict")
            st.image(annotated_np, use_container_width=True)

        st.divider()
        if overall_verdict == "PASS":
            st.success("### Overall Joint Verdict: PASS (Structural Quality Meets Tolerance)")
        else:
            st.error("### Overall Joint Verdict: FAIL (Non-Conformities Detected - Rework Required)")

        if findings:
            st.subheader("Detected Non-Conformities")
            df = pd.DataFrame(findings)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No structural defects or severe workmanship issues detected within highlighted seam.")
