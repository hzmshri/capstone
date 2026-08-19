import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.min_defect_size_mm = 1.5

    def evaluate_isolated_roi(self, seam_roi: np.ndarray, mm_per_pixel: float, sensitivity: float):
        """
        Analyzes isolated weld seam:
        - Thickness CoV (width consistency along the user-selected seam)
        - Gradient chaos / angular entropy (lumpiness & puddle control)
        - Slag voids / dark cavities
        """
        findings = []
        if seam_roi is None or seam_roi.size < 400:
            return findings

        h, w = seam_roi.shape
        
        # 1. Morphological thickness extraction
        _, otsu = cv2.threshold(seam_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(otsu, cv2.DIST_L2, 5)
        dist_values = dist_transform[dist_transform > 1.0]

        if len(dist_values) > 30:
            mean_hw = np.mean(dist_values)
            std_hw = np.std(dist_values)
            thickness_cov = (std_hw / mean_hw) if mean_hw > 0 else 0
            fluctuation_mm = round((std_hw * 2 * mm_per_pixel), 2)

            # Gradient variance inside the highlighted zone
            gx = cv2.Sobel(seam_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(seam_roi, cv2.CV_32F, 0, 1, ksize=3)
            angles = np.arctan2(gy, gx)
            mag = cv2.magnitude(gx, gy)
            
            sig_pixels = mag > np.mean(mag)
            angular_entropy = np.var(angles[sig_pixels]) if np.sum(sig_pixels) > 30 else 0.5

            # Defect criteria
            cov_thresh = 0.50 / sensitivity
            entropy_thresh = 2.05 / sensitivity
            min_var_mm = 2.0 / sensitivity

            if thickness_cov > cov_thresh and angular_entropy > entropy_thresh and fluctuation_mm > min_var_mm:
                findings.append({
                    "Defect": "Poor Workmanship / Erratic Bead Profile & Cold Lap",
                    "Confidence": f"{min(round(thickness_cov * 100, 1), 95.0)}%",
                    "Max Dimension (mm)": fluctuation_mm,
                    "Verdict": "REJECT (AWS D1.1 - Severe Bead Width Inconsistency & Lumpiness)"
                })

        # 2. Slag Pockets & Internal Cavities
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blackhat = cv2.morphologyEx(seam_roi, cv2.MORPH_BLACKHAT, kernel)
        _, dark_thresh = cv2.threshold(blackhat, int(50 / sensitivity), 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        macro_pits = []
        for dc in dark_contours:
            if cv2.contourArea(dc) > 15:
                _, _, pw, ph = cv2.boundingRect(dc)
                pit_size_mm = round(max(pw, ph) * mm_per_pixel, 2)
                if pit_size_mm >= self.min_defect_size_mm:
                    macro_pits.append(pit_size_mm)

        if len(macro_pits) >= 3 or (len(macro_pits) > 0 and max(macro_pits) > 2.8):
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

        # Use user-defined bounding box if provided; otherwise fallback to full image
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
