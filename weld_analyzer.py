import cv2
import numpy as np
from ultralytics import YOLO

class WeldDefectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Uses standard pretrained weights or your custom fine-tuned weights
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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance metallic surface cracks and edge definitions."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def evaluate_compliance(self, label: str, length_mm: float, max_allowable_mm: float = 2.0) -> str:
        """Evaluates defect severity against structural welding limits."""
        if label.lower() in self.critical_defects:
            return "CRITICAL REJECT (Mandatory Gouge & Re-weld)"
        elif length_mm > max_allowable_mm:
            return f"REJECT (Exceeds {max_allowable_mm}mm limit)"
        else:
            return "ACCEPTABLE (Within allowable tolerance)"

    def inspect(self, image_np: np.ndarray, mm_per_pixel: float = 0.05, conf_thresh: float = 0.35):
        enhanced_img = self.preprocess_image(image_np)
        results = self.model.predict(enhanced_img, conf=conf_thresh)
        
        detections = []
        annotated_img = image_np.copy()
        overall_status = "PASS"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.defect_classes.get(cls_id, "defect")

                w_px, h_px = (x2 - x1), (y2 - y1)
                max_dim_mm = round(max(w_px, h_px) * mm_per_pixel, 2)
                
                decision = self.evaluate_compliance(label, max_dim_mm)
                if "REJECT" in decision:
                    overall_status = "FAIL"

                detections.append({
                    "Defect": label.title(),
                    "Confidence": f"{conf:.1%}",
                    "Max Dimension (mm)": max_dim_mm,
                    "Verdict": decision,
                    "BBox": (x1, y1, x2, y2)
                })

                color = (0, 0, 255) if "REJECT" in decision else (0, 255, 255)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} | {max_dim_mm}mm"
                cv2.putText(annotated_img, tag, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_img, detections, overall_status