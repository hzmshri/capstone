import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
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
    img_w, img_h = raw_img.size

    # Scale canvas display for large images to fit UI smoothly
    max_display_w = 700
    display_scale = min(1.0, max_display_w / img_w)
    disp_w = int(img_w * display_scale)
    disp_h = int(img_h * display_scale)
    preview_img = raw_img.resize((disp_w, disp_h))

    st.subheader("Step 1: Highlight Weld Seam")
    st.caption("Drag a rectangle box over the weld bead to exclude all background metal & bolts.")

    # Interactive Drawing Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=preview_img,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key="weld_canvas",
    )

    # Check if a box was drawn
    roi_box = None
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if len(objects) > 0:
            last_rect = objects[-1]
            # Convert canvas coordinates back to original image resolution
            rx = int(last_rect["left"] / display_scale)
            ry = int(last_rect["top"] / display_scale)
            rw = int(last_rect["width"] / display_scale)
            rh = int(last_rect["height"] / display_scale)
            
            # Ensure valid bounds
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(img_w - rx, rw)
            rh = min(img_h - ry, rh)
            
            if rw > 10 and rh > 10:
                roi_box = (rx, ry, rw, rh)

    st.divider()

    # Step 2: Trigger Analysis
    if st.button("Run Joint Analysis", type="primary"):
        raw_np = np.array(raw_img)

        with st.spinner("Analyzing isolated weld seam geometry..."):
            annotated_np, findings, overall_verdict = detector.inspect(
                raw_np,
                user_roi=roi_box,
                mm_per_pixel=mm_per_pixel,
                sensitivity=workmanship_sens
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Capture")
            st.image(raw_img, use_container_width=True)

        with col2:
            st.subheader("QA/QC Annotated Seam")
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
