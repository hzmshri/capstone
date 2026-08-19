import cv2
import numpy as np
from ultralytics import YOLO

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.critical_defects = {"crack", "lack_of_penetration", "incomplete_fusion"}
        self.defect_classes = {
            0: "porosity",
            1: "undercut",
            2: "crack",
            3: "lack_of_penetration",
            4: "spatter",
            5: "excess_reinforcement"
        }

    def analyze_workmanship(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, sensitivity: float = 1.0):
        """
        Segment weld seam and inspect:
        - Bead width consistency (Coefficient of Variation)
        - Surface spatter count & cluster size
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Preprocessing: Bilateral filter removes grain while keeping weld bead edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding to segment distinct metallic features
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        workmanship_flags = []
        annotated_features = image_np.copy()

        if not contours:
            return workmanship_flags, annotated_features

        # Filter contours by area: find candidate weld seam vs small spatter
        contour_areas = [cv2.contourArea(c) for c in contours]
        max_area = max(contour_areas) if contour_areas else 0

        # --- 1. SPATTER & SLAG DETECTION ---
        # Spatter appears as small, high-contrast distinct circular/elliptical dots
        spatter_count = 0
        max_spatter_size_px = 0
        min_spatter_area = 15 * sensitivity
        max_spatter_area = (h * w) * 0.015  # Spatter shouldn't be larger than 1.5% of total image

        for c in contours:
            area = cv2.contourArea(c)
            if min_spatter_area < area < max_spatter_area:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # High circularity or small compact blobs = spatter
                if circularity > 0.4:
                    spatter_count += 1
                    x, y, cw, ch = cv2.boundingRect(c)
                    max_spatter_size_px = max(max_spatter_size_px, max(cw, ch))
                    # Draw subtle yellow marker around detected spatter
                    cv2.rectangle(annotated_features, (x, y), (x + cw, y + ch), (255, 200, 0), 1)

        spatter_size_mm = round(max_spatter_size_px * mm_per_pixel, 2)

        # Flag only if significant spatter count or large slag clumps exist
        spatter_limit = int(12 / sensitivity)
        if spatter_count > spatter_limit or spatter_size_mm > (2.5 / sensitivity):
            workmanship_flags.append({
                "Defect": "Excessive Spatter / Slag Inclusions",
                "Confidence": f"{min(75 + spatter_count * 2, 98)}%",
                "Max Dimension (mm)": spatter_size_mm,
                "Verdict": "REJECT (AWS D1.1 Cl. 6.9 - Surface Cleaning Required)"
            })

        # --- 2. BEAD WIDTH & PROFILE UNIFORMITY ---
        # Find the major weld seam contour (largest structural component)
        valid_seams = [c for c in contours if cv2.contourArea(c) > (h * w * 0.04)]
        
        if valid_seams:
            main_seam = max(valid_seams, key=cv2.contourArea)
            rect = cv2.minAreaRect(main_seam)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Sample cross-sectional widths along the seam
            x, y, rw, rh = cv2.boundingRect(main_seam)
            mask_seam = np.zeros_like(gray)
            cv2.drawContours(mask_seam, [main_seam], -1, 255, -1)

            # Measure slice widths across the vertical or horizontal axis
            slices = []
            if rw > rh:  # Horizontal weld
                step = max(int(rw / 15), 1)
                for col in range(x, x + rw, step):
                    column_pixels = np.count_nonzero(mask_seam[:, col])
                    if column_pixels > 0:
                        slices.append(column_pixels)
            else:  # Vertical weld
                step = max(int(rh / 15), 1)
                for row in range(y, y + rh, step):
                    row_pixels = np.count_nonzero(mask_seam[row, :])
                    if row_pixels > 0:
                        slices.append(row_pixels)

            if len(slices) >= 5:
                mean_width = np.mean(slices)
                std_width = np.std(slices)
                cov = (std_width / mean_width) if mean_width > 0 else 0  # Coefficient of variation

                # A steady, clean weld usually has CoV < 0.28. Irregular/wavy beads exceed 0.35.
                cov_threshold = 0.35 * (1.0 / sensitivity)
                if cov > cov_threshold:
                    variation_mm = round((std_width * mm_per_pixel) * 2, 2)
                    cv2.drawContours(annotated_features, [box], 0, (0, 0, 255), 2)
                    workmanship_flags.append({
                        "Defect": "Irregular Bead Width / Travel Speed Flaw",
                        "Confidence": f"{min(round(cov * 180, 1), 95.0)}%",
                        "Max Dimension (mm)": variation_mm,
                        "Verdict": "REJECT (Non-uniform weld progression)"
                    })
                else:
                    # Clean weld seam contour marked in green
                    cv2.drawContours(annotated_features, [box], 0, (0, 255, 0), 2)

        return workmanship_flags, annotated_features

    def inspect(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, conf_thresh: float = 0.40, sensitivity: float = 1.0):
        detections = []
        annotated_img = image_np.copy()
        overall_status = "PASS"

        # 1. Run Geometric & Workmanship Analysis
        workmanship_findings, seam_annotated = self.analyze_workmanship(image_np, mm_per_pixel, sensitivity)
        detections.extend(workmanship_findings)

        # 2. Run Object Detection Model
        results = self.model.predict(image_np, conf=conf_thresh, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.defect_classes.get(cls_id, "Weld Flaw")

                w_px, h_px = (x2 - x1), (y2 - y1)
                max_dim_mm = round(max(w_px, h_px) * mm_per_pixel, 2)

                decision = "CRITICAL REJECT" if label in self.critical_defects else "REJECT (Out of Spec)"
                detections.append({
                    "Defect": label.title(),
                    "Confidence": f"{conf:.1%}",
                    "Max Dimension (mm)": max_dim_mm,
                    "Verdict": decision
                })

                cv2.rectangle(seam_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(seam_annotated, f"{label}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Evaluate final verdict
        for finding in detections:
            if "REJECT" in finding["Verdict"]:
                overall_status = "FAIL"
                break

        return seam_annotated, detections, overall_status
