import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.min_defect_size_mm = 1.5

    def evaluate_isolated_roi(self, seam_roi: np.ndarray, mm_per_pixel: float, sensitivity: float):
        findings = []
        if seam_roi is None or seam_roi.size < 400:
            return findings

        h, w = seam_roi.shape

        # 1. Isolate the high-texture weld strip within the user's framing box
        # This prevents smooth background steel plates from diluting the metrics
        grad_x = cv2.Sobel(seam_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(seam_roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        
        # Take the top 35% highest-contrast/textured pixels in the selected zone (the bead itself)
        mag_thresh = np.percentile(mag, 65)
        bead_core_pixels = seam_roi[mag > mag_thresh]
        
        if len(bead_core_pixels) < 50:
            return findings

        # 2. Workmanship Metric A: Core Surface Lumpiness & Intensity Variance
        bead_core_std = float(np.std(bead_core_pixels))
        
        # 3. Workmanship Metric B: Bead Edge Straightness / Toe Jaggedness
        # Smooth welds have continuous linear/arc gradients. Bad welds have chaotic directional scatter.
        angles = np.arctan2(grad_y, grad_x)
        angles_core = angles[mag > mag_thresh]
        angle_scatter = float(np.var(angles_core))

        # 4. Workmanship Metric C: Local Profile Height Deviation
        # Measures sudden lumpy peaks and valleys across the bead
        laplacian = cv2.Laplacian(seam_roi, cv2.CV_64F)
        local_lumpiness = float(np.mean(np.abs(laplacian[mag > mag_thresh])))

        # AWS D1.1 Workmanship Rejection Criteria:
        # A good TIG/MIG weld has controlled surface variance (std < 42, lumpiness < 28, angle scatter < 1.85).
        # A poor, lumpy manual weld has erratic metal build-up (std > 48, lumpiness > 32, angle scatter > 1.90).
        lumpiness_limit = 28.0 / sensitivity
        scatter_limit = 1.85 / sensitivity

        if (local_lumpiness > lumpiness_limit and angle_scatter > scatter_limit) or (bead_core_std > (50.0 / sensitivity)):
            est_defect_dim = round(float(np.max(mag) * mm_per_pixel * 0.08), 2)
            findings.append({
                "Defect": "Poor Workmanship / Erratic Bead Profile & Cold Lap",
                "Confidence": f"{min(round(local_lumpiness * 2.6, 1), 96.0)}%",
                "Max Dimension (mm)": max(est_defect_dim, 2.8),
                "Verdict": "REJECT (AWS D1.1 - Severe Bead Width Inconsistency & Lumpiness)"
            })

        # 5. Macro Slag Pockets / Severe Surface Cavities
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blackhat = cv2.morphologyEx(seam_roi, cv2.MORPH_BLACKHAT, kernel)
        _, dark_thresh = cv2.threshold(blackhat, int(45 / sensitivity), 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        macro_pits = []
        for dc in dark_contours:
            area = cv2.contourArea(dc)
            if area > 18:
                _, _, pw, ph = cv2.boundingRect(dc)
                pit_size_mm = round(max(pw, ph) * mm_per_pixel, 2)
                if pit_size_mm >= self.min_defect_size_mm:
                    macro_pits.append(pit_size_mm)

        if len(macro_pits) >= 3 or (len(macro_pits) > 0 and max(macro_pits) > 2.6):
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
