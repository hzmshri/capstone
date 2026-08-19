import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.max_isolated_pore_dia_mm = 2.0

    def classify_joint_geometry(self, contour, image_shape):
        """
        Differentiates between Straight/Linear welds and Rounded/Circumferential pipe welds.
        Returns: 'CIRCULAR_PIPE', 'LINEAR_PLATE', or 'UNKNOWN'
        """
        if len(contour) < 5:
            return "LINEAR_PLATE", 0.0

        # Fit minimum area bounding box
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        major_axis = max(rw, rh)
        minor_axis = max(min(rw, rh), 1)
        aspect_ratio = major_axis / minor_axis

        # Fit ellipse to test circular curvature
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (d1, d2), angle = ellipse
            ellipse_ratio = max(d1, d2) / max(min(d1, d2), 1)
        except Exception:
            ellipse_ratio = 10.0

        # Measure convex hull solidity
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)

        # Circumferential / curved pipe welds show arc curvature (lower solidity, elliptical fit)
        if solidity < 0.65 and ellipse_ratio < 3.2:
            return "CIRCULAR_PIPE", round(solidity, 2)
        else:
            return "LINEAR_PLATE", round(aspect_ratio, 2)

    def isolate_weld_seam(self, gray: np.ndarray):
        """Locates the primary weld seam contour and bounding box."""
        h, w = gray.shape
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        texture_energy = cv2.absdiff(gray, blur)
        
        _, mask = cv2.threshold(texture_energy, 14, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (0, 0, w, h), "LINEAR_PLATE"

        valid_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.01)]
        if not valid_contours:
            main_contour = max(contours, key=cv2.contourArea)
        else:
            main_contour = max(valid_contours, key=cv2.contourArea)

        joint_type, metric = self.classify_joint_geometry(main_contour, gray.shape)
        x, y, bw, bh = cv2.boundingRect(main_contour)
        
        pad = 12
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        return gray[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1), joint_type

    def evaluate_quality(self, seam_roi: np.ndarray, joint_type: str, mm_per_pixel: float, sensitivity: float):
        findings = []
        if seam_roi is None or seam_roi.size == 0:
            return findings

        # 1. Surface Chaos & Roughness Analysis
        laplacian = cv2.Laplacian(seam_roi, cv2.CV_64F)
        roughness_score = laplacian.var()

        # Circumferential TIG pipe welds have tighter periodic ripple textures, so threshold is calibrated by joint type
        if joint_type == "CIRCULAR_PIPE":
            roughness_threshold = 420.0 / sensitivity
        else:
            roughness_threshold = 270.0 / sensitivity

        if roughness_score > roughness_threshold:
            findings.append({
                "Defect": "Poor Workmanship / Erratic Bead Texture",
                "Joint Geometry": joint_type.replace("_", " ").title(),
                "Confidence": f"{min(round(roughness_score / 4.5, 1), 95.0)}%",
                "Max Dimension (mm)": round((seam_roi.shape[0] * mm_per_pixel) * 0.35, 2),
                "Verdict": "REJECT (Excessive Surface Roughness / Unsteady Arc Control)"
            })

        # 2. Slag Pockets & Severe Cavity Detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(seam_roi, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, int(42 / sensitivity), 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitting_count = 0
        max_pit_dia = 0

        for c in contours:
            area = cv2.contourArea(c)
            if 10 < area < (seam_roi.size * 0.03):
                _, _, bw, bh = cv2.boundingRect(c)
                pit_size_mm = round(max(bw, bh) * mm_per_pixel, 2)
                max_pit_dia = max(max_pit_dia, pit_size_mm)
                pitting_count += 1

        if pitting_count >= 3 or max_pit_dia > 2.5:
            findings.append({
                "Defect": "Slag Inclusions / Surface Cavities",
                "Joint Geometry": joint_type.replace("_", " ").title(),
                "Confidence": f"{min(75 + pitting_count * 5, 96)}%",
                "Max Dimension (mm)": max_pit_dia,
                "Verdict": "REJECT (Surface Pockets / Slag Accumulation)"
            })

        return findings

    def inspect(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, conf_thresh: float = 0.45, sensitivity: float = 1.0):
        annotated_img = image_np.copy()
        raw_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # 1. Seam localization and automatic geometry type classification
        seam_roi, bbox, joint_type = self.isolate_weld_seam(raw_gray)
        x, y, w, h = bbox

        # 2. Geometry-specific quality analysis
        findings = self.evaluate_quality(seam_roi, joint_type, mm_per_pixel, sensitivity)
        overall_status = "PASS"

        # 3. Visual Annotations
        geo_label = f"Type: {joint_type.replace('_', ' ')}"
        
        if findings:
            overall_status = "FAIL"
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated_img, f"FAIL | {geo_label}", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"PASS | {geo_label}", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_img, findings, overall_status
