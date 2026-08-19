import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from weld_analyzer import WeldDefectDetector

st.set_page_config(page_title="WeldQC - AI Inspection System", layout="wide")

st.title("Weld Defect Detection & Structural QA/QC System")
st.markdown("Automated vision inspection for steel joint compliance (AWS D1.1 / EN ISO 5817).")

st.sidebar.header("Inspection Parameters")
confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
mm_per_pixel = st.sidebar.number_input("Optical Scale Calibration (mm/px)", value=0.050, format="%.4f")
max_tolerable_len = st.sidebar.number_input("Max Tolerable Minor Flaw (mm)", value=2.0, step=0.5)

detector = WeldDefectDetector()

uploaded_file = st.file_uploader("Upload Steel Weld Joint Capture (JPG / PNG / BMP)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_img)

    with st.spinner("Running deep defect analysis..."):
        annotated_np, findings, overall_verdict = detector.inspect(
            raw_np, mm_per_pixel=mm_per_pixel, conf_thresh=confidence_thresh
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Capture")
        st.image(raw_img, use_container_width=True)

    with col2:
        st.subheader("Annotated Defects")
        st.image(annotated_np, use_container_width=True)

    st.divider()
    if overall_verdict == "PASS":
        st.success("### Overall Joint Verdict: PASS (Ready for Final QC Sign-off)")
    else:
        st.error("### Overall Joint Verdict: FAIL (Rework / Rectification Required)")

    if findings:
        st.subheader("Detected Non-Conformities")
        df = pd.DataFrame(findings).drop(columns=["BBox"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No surface defects detected within current threshold parameters.")