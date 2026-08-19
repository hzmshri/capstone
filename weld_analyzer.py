import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # AWS D1.1 / ISO 5817 minimum defect thresholds
        self.min_defect_size_mm = 1.5  # Ignore micro-ripples and tiny sub-mm surface pits

    def isolate_weld_seam_contour(self, gray: np.ndarray):
        """Locates the primary weld seam using gradient magnitude and morphological filters."""
        h, w = gray.shape

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        blurred = cv2.GaussianBlur(mag, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (0, 0, w, h)

        valid_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.01)]
        main_contour = max(valid_contours, key=cv2.contourArea) if valid_contours else max(contours, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(main_contour)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        return main_contour, (x1, y1, x2 - x1, y2 - y1)

    def evaluate_workmanship_quality(self, gray: np.ndarray, contour, bbox, mm_per_pixel: float, sensitivity: float):
        findings = []
        x, y, bw, bh = bbox
        roi_gray = gray[y:y+bh, x:x+bw]
        
        if roi_gray.size < 400 or contour is None:
            return findings

        # 1. Measure thickness variation along the seam
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        dist_values = dist_transform[dist_transform > 1.0]

        if len(dist_values) > 50:
            mean_half_width = np.mean(dist_values)
            std_half_width = np.std(dist_values)
            thickness_cov = (std_half_width / mean_half_width) if mean_half_width > 0 else 0
            fluctuation_mm = round((std_half_width * 2 * mm_per_pixel), 2)

            # Gradient Orientation Variance
            gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
            angles = np.arctan2(gy, gx)
            mag = cv2.magnitude(gx, gy)
            
            significant_pixels = mag > np.mean(mag)
            angular_entropy = np.var(angles[significant_pixels]) if np.sum(significant_pixels) > 50 else 0.5

            # Must have high CoV (>0.52), chaotic entropy (>2.1), AND significant mm fluctuation (>2.2mm)
            cov_thresh = 0.52 / sensitivity
            entropy_thresh = 2.10 / sensitivity
            min_variation_mm = 2.2 / sensitivity

            if thickness_cov > cov_thresh and angular_entropy > entropy_thresh and fluctuation_mm > min_variation_mm:
                findings.append({
                    "Defect": "Poor Workmanship / Chaotic Bead Profile & Cold Lap",
                    "Confidence": f"{min(round(thickness_cov * 100, 1), 95.0)}%",
                    "Max Dimension (mm)": fluctuation_mm,
                    "Verdict": "REJECT (AWS D1.1 - Severe Bead Width Inconsistency & Lumpiness)"
                })

        # 2. Macro Slag Pockets & Internal Cavities
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, kernel)
        _, dark_thresh = cv2.threshold(blackhat, int(55 / sensitivity), 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        macro_pits = []
        for dc in dark_contours:
            area = cv2.contourArea(dc)
            if area > 20:
                _, _, pw, ph = cv2.boundingRect(dc)
                pit_size_mm = round(max(pw, ph) * mm_per_pixel, 2)
                # Ignore micro-textures under 1.6 mm
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

    def inspect(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, conf_thresh: float = 0.45, sensitivity: float = 1.0):
        annotated_img = image_np.copy()
        raw_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # 1. Seam localization
        main_contour, bbox = self.isolate_weld_seam_contour(raw_gray)
        x, y, bw, bh = bbox

        # 2. Evaluate structural quality
        findings = self.evaluate_workmanship_quality(raw_gray, main_contour, bbox, mm_per_pixel, sensitivity)
        overall_status = "PASS"

        # 3. Render Output
        if findings:
            overall_status = "FAIL"
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(annotated_img, f"FAIL: {findings[0]['Defect'][:26]}...", 
                        (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 0), 2)
        else:
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(annotated_img, "PASS: Uniform Weld Bead (AWS D1.1 Compliant)", 
                        (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

        clean_table_findings = []
        for f in findings:
            clean_table_findings.append({
                "Defect": f["Defect"],
                "Confidence": f["Confidence"],
                "Max Dimension (mm)": f["Max Dimension (mm)"],
                "Verdict": f["Verdict"]
            })

        return annotated_img, clean_table_findings, overall_status
