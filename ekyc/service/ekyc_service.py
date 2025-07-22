import shutil
import yolov5
import ekyc.utils.utils as utils
import cv2 as cv
import re
import os
import torch
import numpy as np
import ultralytics

from PIL import Image
from facenet_pytorch import MTCNN
from ekyc.service.face_check_service.face_matching_service import FaceVerifier
from ekyc.service.face_check_service.blink_detection import BlinkDetector
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from ekyc.common.functions import get_image
from io import BytesIO


class EKYCService:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = MTCNN(device=self.device)
        self.blink_detector = BlinkDetector()
        self.verifier = FaceVerifier(device=self.device)

        self.CONF_CONTENT_THRESHOLD = 0.7
        self.CORNER_MODEL = ultralytics.YOLO("/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/corner.pt")
        self.CORNER_BACK = yolov5.load("/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/corner_back.pt")
        self.CONTENT_MODEL = yolov5.load("/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/content.pt")
        self.CONTENT_BACK = ultralytics.YOLO("/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/content_back.pt")
        self.NEW_CONTENT_BACK = ultralytics.YOLO("/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/new_content_back.pt")
        self.CONTENT_MODEL.conf = self.CONF_CONTENT_THRESHOLD

        self.config = Cfg.load_config_from_name("vgg_seq2seq")
        self.config["cnn"]["pretrained"] = False
        self.config["device"] = self.device
        self.config["predictor"]["beamsearch"] = False
        self.ocr_detector = Predictor(self.config)

    def process_ekyc(self, video_path, cccd_path, max_duration=7):

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Không thể mở video."}

        fps = int(cap.get(cv.CAP_PROP_FPS))
        max_frames = fps * max_duration
        blink_detected = False
        best_face, max_box_area = None, 0

        frame_count = 0
        while frame_count <= max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 != 0:
                frame_count += 1
                continue

            small_frame = cv.resize(frame, (640, 360))
            rgb = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

            boxes, probs, landmarks = self.detector.detect(rgb, landmarks=True)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].astype(int)
                box_area = (x2 - x1) * (y2 - y1)

                if not blink_detected:
                    blink_detected = self.blink_detector.eye_blink(rgb, [x1, y1, x2, y2])

                if box_area > max_box_area:
                    scale_x = frame.shape[1] / 640
                    scale_y = frame.shape[0] / 360
                    full_x1, full_y1 = int(x1 * scale_x), int(y1 * scale_y)
                    full_x2, full_y2 = int(x2 * scale_x), int(y2 * scale_y)
                    best_face = frame[full_y1:full_y2, full_x1:full_x2]
                    max_box_area = box_area

                if blink_detected:
                    break
            frame_count += 1

        cap.release()

        if not blink_detected:
            return {"success": False, "error": "Eyes detection failed"}
        if best_face is None or best_face.size == 0:
            return {"success": False, "error": "Facial detection failed"}

        best_face_rgb = cv.cvtColor(best_face, cv.COLOR_BGR2RGB)
        cccd_image = get_image(cccd_path)
        match = self.verifier.verify(best_face_rgb, cccd_image)

        return {
            "liveness": blink_detected,
            "face_matching": bool(match)
        }

    def process_ocr_back(self, image_path):
        image = cv.imread(image_path)
        if image is None:
            return {"success": False, "error": f"Không đọc được ảnh từ {image_path}"}

        # --- Step 1: Model mới ---
        new_results = self.NEW_CONTENT_BACK(image, verbose=False)[0]
        new_class_names = {
            0: 'Date of expirty', 1: 'Date of issue', 2: 'Place', 3: 'Place of birth',
            4: 'address_1', 5: 'address_2', 6: 'bottom_left', 7: 'bottom_right',
            8: 'cdate_of_birth', 9: 'cdate_of_expiry', 10: 'cdate_of_issue',
            11: 'cplace_of_birth', 12: 'date_of_expiry', 13: 'date_of_issue',
            14: 'place', 15: 'place_of_birth', 16: 'top_left', 17: 'top_right'
        }

        extracted_new = {}
        found_issue_date = False

        for box in new_results.boxes:
            conf = box.conf[0].item()
            if conf < 0.6:
                continue

            cls_id = int(box.cls[0].item())
            label = new_class_names.get(cls_id, f"class_{cls_id}")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            text = self.ocr_detector.predict(crop_pil).strip()
            if text and len(text) >= 3:
                extracted_new[label] = text
                if label in ['cdate_of_issue', 'Date of issue']:
                    found_issue_date = True

        # Trích lọc và chuẩn hoá dữ liệu model mới (nếu hợp lệ)
        if extracted_new and found_issue_date:
            keys_of_interest = [
                'address_1',
                'address_2',
                'cdate_of_issue',
                'cdate_of_expiry',
                'cplace_of_birth'
            ]
            simplified_result = {}

            for key in keys_of_interest:
                raw = extracted_new.get(key, "")
                if ":" in raw:
                    simplified_result[key] = raw.split(":", 1)[-1].strip()
                else:
                    simplified_result[key] = raw

            return {
                "success": True,
                "version": "new",
                "data": {
                    "address": simplified_result.get("address_1", "") + " " + simplified_result.get("address_2", ""),
                    "issue_date": simplified_result.get("cdate_of_issue", ""),
                    "expiry_date": simplified_result.get("cdate_of_expiry", ""),
                    "place_of_birth": simplified_result.get("cplace_of_birth", "")
                }
            }

        # --- Step 2: fallback sang model cũ ---
        old_results = self.CONTENT_BACK(image, verbose=False)[0]
        old_class_names = {
            0: 'Issue_date',
            1: 'Issuer',
            2: 'MRZ',
            3: 'Personal_identification',
            4: 'fingerprint'
        }

        for box in old_results.boxes:
            conf = box.conf[0].item()
            if conf < 0.6:
                continue

            cls_id = int(box.cls[0].item())
            label = old_class_names.get(cls_id, f"class_{cls_id}")

            if label == "Issue_date":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]
                crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                text = self.ocr_detector.predict(crop_pil).strip()
                date_match = re.search(r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})', text)
                issue_date = date_match.group(1) if date_match else text
                return {
                    "success": True,
                    "version": "old",
                    "data": {
                        "init_date": issue_date
                    }
                }

        return {
            "success": False,
            "error": "Không phát hiện được thông tin mặt sau từ cả 2 mô hình (cũ/mới)."
        }

    def detect_corner(self, image_path: str):

        img = Image.open(image_path).convert("RGB")

        corner_preds = self.CORNER_BACK(image_path).pred[0]
        if corner_preds is None or corner_preds.size(0) < 4:
            raise ValueError("Không phát hiện đủ 4 góc.")

        corner_boxes = corner_preds[:, :4].tolist()
        corner_classes = corner_preds[:, 5].tolist()
        if len(corner_classes) != 4:
            raise ValueError("Không phát hiện đủ 4 góc.")

        ordered_boxes = utils.class_Order(corner_boxes, corner_classes)
        center_points = list(map(utils.get_center_point, ordered_boxes))

        c2, c3 = center_points[2], center_points[3]
        c2 = (c2[0], c2[1] + 30)
        c3 = (c3[0], c3[1] + 30)
        center_points = [center_points[0], center_points[1], c2, c3]

        aligned = utils.four_point_transform(img, np.asarray(center_points))
        aligned_img = Image.fromarray(aligned)
        return aligned_img

    def process_ocr(self, image_path):
        try:
            aligned_img = self.detect_corner(image_path)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        if isinstance(aligned_img, dict) and aligned_img.get("success") is False:
            return aligned_img  # Trả lỗi từ detect_corner ra luôn

        content_preds = self.CONTENT_MODEL(aligned_img).pred[0]
        if content_preds is None or content_preds.size(0) == 0:
            return {"success": False, "error": "Không phát hiện nội dung."}

        content_boxes = content_preds[:, :4].tolist()
        content_classes = content_preds[:, 5].tolist()

        boxes, classes = utils.non_max_suppression_fast(np.array(content_boxes), content_classes, 0.7)
        boxes = utils.class_Order(boxes, classes)

        fields = []
        for i, box in enumerate(boxes):
            left, top, right, bottom = map(int, box)
            if 5 < i < 9:
                right += 100
            cropped = aligned_img.crop((left, top, right, bottom))
            if i > 0:
                text = self.ocr_detector.predict(cropped)
                fields.append(text)

        if 7 in content_classes and len(fields) >= 9:
            fields = fields[:6] + [fields[6] + ", " + fields[7]] + [fields[8]]

        version = "unknown"
        if len(fields) >= 7:
            version = "old"
            field_names = ["idNumber", "fullName", "dob", "gender", "nationality", "birthplace", "address"]
        elif len(fields) <= 6:
            version = "new"
            field_names = ["idNumber", "fullName", "gender", "nationality", "dob"]
        else:
            return {"success": False, "error": f"Không thể xác định phiên bản CCCD. Số field: {fields}"}

        result = {k: v for k, v in zip(field_names, fields)}
        return result