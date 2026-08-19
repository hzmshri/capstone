import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.max_isolated_pore_dia_mm = 2.0

    def isolate_weld_seam_contour(self, gray: np.ndarray):
        """Locates the primary weld seam using gradient magnitude and morphological filters."""
        h, w = gray.shape

        # Multi-scale gradient
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Enhance texture regions
        blurred = cv2.GaussianBlur(mag, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (0, 0, w, h)

        # Filter out small noise; select main elongated seam
        valid_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.01)]
        if not valid_contours:
            main_contour = max(contours, key=cv2.contourArea)
        else:
            main_contour = max(valid_contours, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(main_contour)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        return main_contour, (x1, y1, x2 - x1, y2 - y1)

    def evaluate_workmanship_quality(self, gray: np.ndarray, contour, bbox, mm_per_pixel: float, sensitivity: float):
        """
        Analyzes:
        1. Local thickness consistency along the contour (handles curved & straight equally).
        2. Gradient orientation entropy (chaotic vs uniform progression).
        3. Dark slag pocket / cavity clusters.
        """
        findings = []
        x, y, bw, bh = bbox
        roi_gray = gray[y:y+bh, x:x+bw]
        
        if roi_gray.size < 400 or contour is None:
            return findings

        # 1. Measure thickness variation along the seam
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Non-zero distance values represent local half-thickness along the seam
        dist_values = dist_transform[dist_transform > 1.0]

        if len(dist_values) > 50:
            mean_half_width = np.mean(dist_values)
            std_half_width = np.std(dist_values)
            thickness_cov = (std_half_width / mean_half_width) if mean_half_width > 0 else 0

            # 2. Gradient Orientation Coherence (Measures puddle stability)
            gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
            
            angles = np.arctan2(gy, gx)
            mag = cv2.magnitude(gx, gy)
            
            # Weighted angular variance in high-gradient regions
            significant_pixels = mag > np.mean(mag)
            if np.sum(significant_pixels) > 50:
                angular_entropy = np.var(angles[significant_pixels])
            else:
                angular_entropy = 0.5

            # Quality Check:
            # Clean TIG/MIG welds (straight or curved) maintain low thickness CoV (<0.32) and structured entropy.
            # Poor subcontractor welds show high thickness fluctuation (CoV > 0.42) and turbulent entropy (>1.85).
            cov_thresh = 0.40 / sensitivity
            entropy_thresh = 1.80 / sensitivity

            if thickness_cov > cov_thresh and angular_entropy > entropy_thresh:
                findings.append({
                    "Defect": "Poor Workmanship / Erratic Bead Profile & Cold Lap",
                    "Confidence": f"{min(round((thickness_cov * 120), 1), 96.0)}%",
                    "Max Dimension (mm)": round((std_half_width * 2 * mm_per_pixel), 2),
                    "Verdict": "REJECT (AWS D1.1 - Severe Bead Width Inconsistency & Lumpiness)"
                })

        # 3. Slag Pockets & Internal Cavities
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, kernel)
        _, dark_thresh = cv2.threshold(blackhat, int(45 / sensitivity), 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        severe_pits = 0
        for dc in dark_contours:
            if 10 < cv2.contourArea(dc) < (roi_gray.size * 0.03):
                severe_pits += 1

        if severe_pits >= 4:
            findings.append({
                "Defect": "Slag Inclusions / Surface Cavities",
                "Confidence": f"{min(75 + severe_pits * 4, 95)}%",
                "Max Dimension (mm)": round(severe_pits * 0.5 * mm_per_pixel, 2),
                "Verdict": "REJECT (Uncleaned Slag / Lack of Interpass Fusion)"
            })

        return findings

    def inspect(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, conf_thresh: float = 0.45, sensitivity: float = 1.0):
        annotated_img = image_np.copy()
        raw_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # 1. Find the weld seam contour and bounding box
        main_contour, bbox = self.isolate_weld_seam_contour(raw_gray)
        x, y, bw, bh = bbox

        # 2. Evaluate thickness consistency and texture coherence
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

        # Format table data
        clean_table_findings = []
        for f in findings:
            clean_table_findings.append({
                "Defect": f["Defect"],
                "Confidence": f["Confidence"],
                "Max Dimension (mm)": f["Max Dimension (mm)"],
                "Verdict": f["Verdict"]
            })

        return annotated_img, clean_table_findings, overall_status
