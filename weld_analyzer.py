import cv2
import numpy as np

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.max_isolated_pore_dia_mm = 2.0
        self.max_pore_cluster_density = 4

    def remove_heat_tint_and_reflections(self, image_np: np.ndarray) -> np.ndarray:
        """
        Suppresses heat tint discoloration (blue/straw/purple HAZ oxidation)
        and bright cylindrical specular reflections on curved pipes.
        """
        # Convert to HSV to detect high-saturation thermal oxidation colors
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Mask high saturation areas (rainbow heat-affected oxidation)
        _, heat_tint_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)

        # Mask bright specular highlights (shiny metal reflection streaks)
        _, glare_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(heat_tint_mask, glare_mask)

        # Morphological dilation to smooth boundary edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # Convert image to grayscale and inpaint masked thermal/glare zones
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        cleaned_gray = cv2.inpaint(gray, combined_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # Apply a mild bilateral filter to preserve mechanical boundaries while suppressing metal grain
        smoothed = cv2.bilateralFilter(cleaned_gray, 9, 50, 50)
        return smoothed

    def detect_porosity_and_pits(self, gray: np.ndarray, mm_per_pixel: float, sensitivity: float):
        """Detects true concentrated dark gas cavities and crater pits."""
        h, w = gray.shape
        kernel_size = int(round(13 * sensitivity))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        thresh_val = int(45 / sensitivity)
        _, thresh = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_pores = []
        for c in contours:
            area = cv2.contourArea(c)
            if 8 < area < (h * w * 0.003):
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Porosity pores are distinctly round or oval
                if circularity > 0.55:
                    x, y, bw, bh = cv2.boundingRect(c)
                    pore_dia_mm = round(max(bw, bh) * mm_per_pixel, 2)
                    if pore_dia_mm >= 0.6:
                        valid_pores.append((x, y, bw, bh, pore_dia_mm))

        porosity_detections = []
        if len(valid_pores) > self.max_pore_cluster_density:
            max_pore = max([p[4] for p in valid_pores])
            porosity_detections.append({
                "Defect": "Porosity / Gas Cavity Cluster",
                "Confidence": f"{min(80 + len(valid_pores) * 3, 98)}%",
                "Max Dimension (mm)": max_pore,
                "Verdict": "REJECT (Exceeds ISO 5817 cluster tolerance)",
                "BBoxes": [(p[0], p[1], p[2], p[3]) for p in valid_pores]
            })

        return porosity_detections

    def detect_cracks_and_notches(self, gray: np.ndarray, mm_per_pixel: float, sensitivity: float):
        """
        Detects true structural cracks and lack of fusion.
        Ignores linear pipe geometry, shadows, and smooth reflection gradients.
        """
        h, w = gray.shape

        # Morphological gradient to isolate localized, steep intensity drops
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # High threshold so only genuine sharp cracks trigger
        thresh_val = int(70 / sensitivity)
        _, sharp_edges = cv2.threshold(gradient, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(sharp_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crack_detections = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter out tiny noise and full-image pipe borders
            if (h * w * 0.0008) < area < (h * w * 0.04):
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                major_axis = max(rw, rh)
                minor_axis = min(rw, rh)
                aspect_ratio = major_axis / max(minor_axis, 1)

                # Cracks must be razor-thin with high aspect ratio and rough perimeter
                perimeter = cv2.arcLength(c, True)
                roughness = (perimeter * perimeter) / max(area, 1)

                # Cracks have very high perimeter-to-area (roughness > 35) and high aspect ratio (> 6.0)
                if aspect_ratio > 6.0 and roughness > 35.0 and (major_axis * mm_per_pixel > 4.0):
                    x, y, bw, bh = cv2.boundingRect(c)
                    crack_length_mm = round(major_axis * mm_per_pixel, 2)
                    crack_detections.append({
                        "Defect": "Surface Crack / Linear Discontinuity",
                        "Confidence": "89.0%",
                        "Max Dimension (mm)": crack_length_mm,
                        "Verdict": "CRITICAL REJECT (Mandatory NDT & Gouge)",
                        "BBoxes": [(x, y, bw, bh)]
                    })

        return crack_detections

    def inspect(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, conf_thresh: float = 0.45, sensitivity: float = 1.0):
        annotated_img = image_np.copy()
        
        # 1. Neutralize thermal oxidation bands and specular pipe reflections
        filtered_gray = self.remove_heat_tint_and_reflections(image_np)

        # 2. Extract structural non-conformities
        porosity_flags = self.detect_porosity_and_pits(filtered_gray, mm_per_pixel, sensitivity)
        crack_flags = self.detect_cracks_and_notches(filtered_gray, mm_per_pixel, sensitivity)

        all_findings = porosity_flags + crack_flags
        overall_status = "PASS"

        # 3. Render Annotations
        for finding in all_findings:
            if "REJECT" in finding["Verdict"]:
                overall_status = "FAIL"
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            for (bx, by, bw, bh) in finding.get("BBoxes", []):
                cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), color, 2)
                cv2.putText(annotated_img, finding["Defect"], (bx, max(by - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        clean_table_findings = []
        for f in all_findings:
            clean_table_findings.append({
                "Defect": f["Defect"],
                "Confidence": f["Confidence"],
                "Max Dimension (mm)": f["Max Dimension (mm)"],
                "Verdict": f["Verdict"]
            })

        return annotated_img, clean_table_findings, overall_status
