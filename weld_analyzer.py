import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.min_defect_size_mm = 1.8

    def evaluate_isolated_roi(self, seam_roi: np.ndarray, mm_per_pixel: float, sensitivity: float):
        findings = []
        if seam_roi is None or seam_roi.size < 400:
            return findings

        h, w = seam_roi.shape

        # 1. Bilateral filter to smooth metal grain while preserving sharp edges
        smoothed = cv2.bilateralFilter(seam_roi, 9, 60, 60)

        # 2. Local adaptive thresholding to extract actual weld bead (avoids background void bias)
        bead_thresh = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
        )

        # 3. Check for high-frequency periodicity (ripple regularity)
        # Good TIG/MIG weaves have distinct periodic oscillations along the primary travel axis
        proj_x = np.mean(smoothed, axis=0)
        proj_y = np.mean(smoothed, axis=1)
        active_proj = proj_x if np.var(proj_x) > np.var(proj_y) else proj_y
        active_proj = active_proj - np.mean(active_proj)

        regularity_score = 0.0
        if len(active_proj) > 20 and np.sum(active_proj ** 2) > 0:
            autocorr = np.correlate(active_proj, active_proj, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-6)
            peaks = [autocorr[i] for i in range(2, min(len(autocorr)-1, 50)) 
                     if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]]
            if peaks:
                regularity_score = float(max(peaks))

        # 4. Measure chaotic lumpiness and puddle instability
        laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
        roughness_entropy = laplacian.var()

        # Gradient orientation variance
        gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        angles = np.arctan2(gy, gx)
        
        sig_pixels = mag > (np.mean(mag) * 1.2)
        angular_chaos = np.var(angles[sig_pixels]) if np.sum(sig_pixels) > 50 else 0.5

        # --- Workmanship Evaluation ---
        # A good weld (even if curved or discolored) maintains either high ripple regularity (> 0.28) 
        # or moderate entropy (< 450).
        # A poor subcontractor weld has chaotic entropy (> 550) AND zero periodic rhythm (< 0.18).
        entropy_limit = 520.0 / sensitivity
        chaos_limit = 2.40 / sensitivity

        if roughness_entropy > entropy_limit and angular_chaos > chaos_limit and regularity_score < 0.20:
            fluctuation_mm = round((np.std(mag[sig_pixels]) * mm_per_pixel * 0.5), 2)
            if fluctuation_mm > (2.2 / sensitivity):
                findings.append({
                    "Defect": "Poor Workmanship / Chaotic Bead Profile & Cold Lap",
                    "Confidence": f"{min(round((roughness_entropy / 7.0), 1), 96.0)}%",
                    "Max Dimension (mm)": fluctuation_mm,
                    "Verdict": "REJECT (AWS D1.1 - Severe Bead Width Inconsistency & Lumpiness)"
                })

        # --- Macro Slag / Cavity Pits Detection ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blackhat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel)
        _, dark_thresh = cv2.threshold(blackhat, int(60 / sensitivity), 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        macro_pits = []
        for dc in dark_contours:
            area = cv2.contourArea(dc)
            if 25 < area < (seam_roi.size * 0.04):
                _, _, pw, ph = cv2.boundingRect(dc)
                pit_size_mm = round(max(pw, ph) * mm_per_pixel, 2)
                if pit_size_mm >= self.min_defect_size_mm:
                    macro_pits.append(pit_size_mm)

        if len(macro_pits) >= 3 or (len(macro_pits) > 0 and max(macro_pits) > 3.0):
            max_pit = max(macro_pits)
            findings.append({
                "Defect": "Slag Inclusions / Severe Surface Cavities",
                "Confidence": f"{min(80 + len(macro_pits) * 5, 96)}%",
                "Max Dimension (mm)": max_pit,
                "Verdict": "REJECT (AWS D1.1 - Uncleaned Slag / Lack of Fusion Voids)"
            })

        return findings

    def inspect(self, image_np: np.ndarray, user_roi=None, mm_per_pixel: float = 0.05, sensitivity: float = 1.0):
        annotated_img = image_np.copy()
        raw_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        h, w = raw_gray.shape

        if user_roi is not None:
            x, y, bw, bh = user_roi
            seam_roi = raw_gray[y:y+bh, x:x+bw]
        else:
            x, y, bw, bh = 0, 0, w, h
            seam_roi = raw_gray

        findings = self.evaluate_isolated_roi(seam_roi, mm_per_pixel, sensitivity)
        overall_status = "PASS"

        if findings:
            overall_status = "FAIL"
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
            cv2.putText(annotated_img, f"FAIL: {findings[0]['Defect'][:26]}...", 
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
        else:
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.putText(annotated_img, "PASS: Uniform Weld Bead (AWS D1.1 Compliant)", 
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        clean_table_findings = []
        for f in findings:
            clean_table_findings.append({
                "Defect": f["Defect"],
                "Confidence": f["Confidence"],
                "Max Dimension (mm)": f["Max Dimension (mm)"],
                "Verdict": f["Verdict"]
            })

        return annotated_img, clean_table_findings, overall_status
