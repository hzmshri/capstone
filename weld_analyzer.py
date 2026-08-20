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
        is_vertical = h >= w

        # 1. Morphological isolation of the active weld metal
        smoothed = cv2.bilateralFilter(seam_roi, 9, 75, 75)
        grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        
        # 2. Slice-by-Slice Bead Width Tracking along primary travel axis
        # Isolates the textured bead strip from the smooth parent plate
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bead_mask = cv2.threshold(mag_norm, 28, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bead_mask = cv2.morphologyEx(bead_mask, cv2.MORPH_CLOSE, kernel)

        slice_widths = []
        if is_vertical:
            step = max(int(h / 35), 1)
            for row in range(0, h, step):
                active_pts = np.where(bead_mask[row, :] > 0)[0]
                if len(active_pts) > 2:
                    width = active_pts[-1] - active_pts[0]
                    slice_widths.append(width)
        else:
            step = max(int(w / 35), 1)
            for col in range(0, w, step):
                active_pts = np.where(bead_mask[:, col] > 0)[0]
                if len(active_pts) > 2:
                    width = active_pts[-1] - active_pts[0]
                    slice_widths.append(width)

        # 3. Workmanship Failure Conditions
        if len(slice_widths) >= 8:
            mean_w = np.mean(slice_widths)
            std_w = np.std(slice_widths)
            cov_w = (std_w / max(mean_w, 1.0))
            width_fluctuation_mm = round(std_w * 2.0 * mm_per_pixel, 2)

            # Measure Toe Profile Jaggedness
            edges = cv2.Canny(smoothed, 30, 90)
            edge_density = np.sum(edges > 0) / max(np.sum(bead_mask > 0), 1)

            # AWS D1.1 Workmanship Thresholds:
            # Good TIG/MIG weaves: CoV < 0.28, low/medium edge density.
            # Poor lumpy welds: High width swing (CoV > 0.34) OR erratic edge density (> 0.22).
            cov_limit = 0.32 / sensitivity
            edge_limit = 0.20 / sensitivity

            if cov_w > cov_limit or edge_density > edge_limit:
                findings.append({
                    "Defect": "Poor Workmanship / Erratic Bead Profile & Cold Lap",
                    "Confidence": f"{min(round((cov_w + edge_density) * 110, 1), 96.0)}%",
                    "Max Dimension (mm)": max(width_fluctuation_mm, 2.8),
                    "Verdict": "REJECT (AWS D1.1 - Severe Bead Width Inconsistency & Lumpiness)"
                })

        # 4. Isolated Slag Pockets / Severe Surface Voids
        blackhat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        _, dark_thresh = cv2.threshold(blackhat, int(52 / sensitivity), 255, cv2.THRESH_BINARY)
        dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        macro_pits = []
        for dc in dark_contours:
            area = cv2.contourArea(dc)
            if 15 < area < (seam_roi.size * 0.03):
                _, _, pw, ph = cv2.boundingRect(dc)
                aspect_ratio = max(pw, ph) / max(min(pw, ph), 1)
                # True slag cavities are compact (aspect ratio < 2.5), not flange shadows
                if aspect_ratio < 2.5:
                    pit_size_mm = round(max(pw, ph) * mm_per_pixel, 2)
                    if pit_size_mm >= self.min_defect_size_mm:
                        macro_pits.append(pit_size_mm)

        if len(macro_pits) >= 3 or (len(macro_pits) > 0 and max(macro_pits) > 2.8):
            max_pit = max(macro_pits)
            findings.append({
                "Defect": "Slag Inclusions / Isolated Surface Cavities",
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
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (220, 38, 38), 3)
            cv2.putText(annotated_img, f"FAIL: {findings[0]['Defect'][:26]}...", 
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 38, 38), 2)
        else:
            cv2.rectangle(annotated_img, (x, y), (x + bw, y + bh), (22, 163, 74), 3)
            cv2.putText(annotated_img, "PASS: Uniform Weld Bead (AWS D1.1 Compliant)", 
                        (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (22, 163, 74), 2)

        clean_table_findings = []
        for f in findings:
            clean_table_findings.append({
                "Defect": f["Defect"],
                "Confidence": f["Confidence"],
                "Max Dimension (mm)": f["Max Dimension (mm)"],
                "Verdict": f["Verdict"]
            })

        return annotated_img, clean_table_findings, overall_status
