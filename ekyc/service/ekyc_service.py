import shutil
import yolov5
import time
from PIL import Image
import numpy as np
import ultralytics
from facenet_pytorch import MTCNN
from ekyc.service.face_check_service.face_matching_service import FaceVerifier
from ekyc.service.face_check_service.blink_detection import BlinkDetector
from ekyc.service.face_check_service.face_orientation import FaceOrientationDetector
from ekyc.config.s3.s3_client import S3Client
from ekyc.repository.user_repository import UserRepository
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import ekyc.utils.utils as utils
import cv2 as cv
import re
import os
import torch
from ekyc.common.functions import get_image


class EKYCService:

    def __init__(self, s3_client: S3Client, user_repository: UserRepository):
        self.s3_client = s3_client
        self.user_repository = user_repository
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = MTCNN(device=self.device)
        self.blink_detector = BlinkDetector()
        self.orientation_detector = FaceOrientationDetector()
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
        start_time = time.time()

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Không thể mở video."}

        fps = int(cap.get(cv.CAP_PROP_FPS))
        max_frames = fps * max_duration
        blink_detected = frontal_detected = left_detected = right_detected = False
        best_face, max_box_area = None, 0

        frame_count = 0
        while frame_count <= max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 6 != 0:
                frame_count += 1
                continue

            small_frame = cv.resize(frame, (640, 360))
            rgb = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

            boxes, probs, landmarks = self.detector.detect(rgb, landmarks=True)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].astype(int)
                box_area = (x2 - x1) * (y2 - y1)
                lm = landmarks[0]

                if not blink_detected:
                    blink_detected = self.blink_detector.eye_blink(rgb, [x1, y1, x2, y2])

                try:
                    orientation = self.orientation_detector.detect(lm)
                    frontal_detected |= orientation == "front"
                    left_detected |= orientation == "left"
                    right_detected |= orientation == "right"
                except:
                    break

                if box_area > max_box_area:
                    scale_x = frame.shape[1] / 640
                    scale_y = frame.shape[0] / 360
                    full_x1, full_y1 = int(x1 * scale_x), int(y1 * scale_y)
                    full_x2, full_y2 = int(x2 * scale_x), int(y2 * scale_y)
                    best_face = frame[full_y1:full_y2, full_x1:full_x2]
                    max_box_area = box_area

                if all([blink_detected, frontal_detected, left_detected, right_detected]):
                    break
            frame_count += 1

        cap.release()
        print(f"[INFO] Liveness & Orientation detection time: {time.time() - start_time:.2f}s")

        if not blink_detected:
            return {"success": False, "error": "Eyes detection failed"}
        if not (frontal_detected and left_detected and right_detected):
            return {"success": False, "error": "Facial Orientation detection failed"}
        if best_face is None or best_face.size == 0:
            return {"success": False, "error": "Facial detection failed"}

        best_face_rgb = cv.cvtColor(best_face, cv.COLOR_BGR2RGB)
        cccd_image = get_image(cccd_path)
        match_start = time.time()
        match = self.verifier.verify(best_face_rgb, cccd_image)
        print(f"[INFO] Face verification time: {time.time() - match_start:.2f}s")

        return {
            "liveness": blink_detected,
            "facial_orientation": all([frontal_detected, left_detected, right_detected]),
            "face_matching": bool(match)
        }

    def process_ocr_back(self, image_path):
        image = cv.imread(image_path)
        if image is None:
            return {"success": False, "error": f"Không đọc được ảnh từ {image_path}"}

        # ==== CCCD MỚI: Detect bằng NEW_CONTENT_BACK ====
        new_results = self.NEW_CONTENT_BACK(image)[0]
        new_labels_of_interest = ["cdate_of_issue", "cdate_of_expiry", "address_1", "address_2"]
        new_class_names = {
            0: 'Date of expirty', 1: 'Date of issue', 2: 'Place', 3: 'Place of birth',
            4: 'address_1', 5: 'address_2', 6: 'bottom_left', 7: 'bottom_right',
            8: 'cdate_of_birth', 9: 'cdate_of_expiry', 10: 'cdate_of_issue',
            11: 'cplace_of_birth', 12: 'date_of_expiry', 13: 'date_of_issue',
            14: 'place', 15: 'place_of_birth', 16: 'top_left', 17: 'top_right'
        }

        extracted_new = {}
        valid_new_detections = []

        for box in new_results.boxes:
            conf = box.conf[0].item()
            if conf < 0.6:
                continue

            cls_id = int(box.cls[0].item())
            label = new_class_names.get(cls_id, f"class_{cls_id}")
            if label in new_labels_of_interest:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]
                crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                text = self.ocr_detector.predict(crop_pil).strip()

                if text and len(text) >= 5:
                    extracted_new[label] = text
                    valid_new_detections.append(label)

        old_results = self.CONTENT_BACK(image)[0]
        old_class_names = {
            0: 'Issue_date',
            1: 'Issuer',
            2: 'MRZ',
            3: 'Personal_identification',
            4: 'fingerprint'
        }

        detected_old_labels = []
        issue_date_result = None

        for box in old_results.boxes:
            conf = box.conf[0].item()
            if conf < 0.6:
                continue

            cls_id = int(box.cls[0].item())
            label = old_class_names.get(cls_id, f"class_{cls_id}")
            detected_old_labels.append(label)

            if label == "Issue_date":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]
                crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                text = self.ocr_detector.predict(crop_pil).strip()
                date_match = re.search(r'(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})', text)
                issue_date_result = date_match.group(1) if date_match else text


        if issue_date_result:
            return issue_date_result

        return {
            "success": False,
            "error": "Không xác định được loại CCCD (cũ/mới) hoặc không phát hiện được vùng cần thiết."
        }

    def process_ocr(self, image_path):
        t_start = time.time()
        img = Image.open(image_path).convert("RGB")

        # Detect corners
        corner_preds = self.CORNER_BACK(image_path).pred[0]
        if corner_preds is None or corner_preds.size(0) < 4:
            return {"success": False, "error": "Không phát hiện đủ 4 góc."}

        corner_boxes = corner_preds[:, :4].tolist()
        corner_classes = corner_preds[:, 5].tolist()
        if len(corner_classes) != 4:
            return {"success": False, "error": "Không phát hiện đủ 4 góc."}

        ordered_boxes = utils.class_Order(corner_boxes, corner_classes)
        center_points = list(map(utils.get_center_point, ordered_boxes))

        c2, c3 = center_points[2], center_points[3]
        c2 = (c2[0], c2[1] + 30)
        c3 = (c3[0], c3[1] + 30)
        center_points = [center_points[0], center_points[1], c2, c3]

        aligned = utils.four_point_transform(img, np.asarray(center_points))
        aligned_img = Image.fromarray(aligned)

        print(f"[INFO] Alignment time: {time.time() - t_start:.2f}s")

        # Content detection
        t_content = time.time()
        content_preds = self.CONTENT_MODEL(aligned_img).pred[0]
        if content_preds is None or content_preds.size(0) == 0:
            return {"success": False, "error": "Không phát hiện nội dung."}

        content_boxes = content_preds[:, :4].tolist()
        content_classes = content_preds[:, 5].tolist()

        boxes, classes = utils.non_max_suppression_fast(np.array(content_boxes), content_classes, 0.7)
        boxes = utils.class_Order(boxes, classes)

        print(f"[INFO] Content detection time: {time.time() - t_content:.2f}s")

        # OCR các vùng
        t_ocr = time.time()
        fields = []
        for i, box in enumerate(boxes):
            left, top, right, bottom = map(int, box)
            if 5 < i < 9:
                right += 100
            cropped = aligned_img.crop((left, top, right, bottom))
            if i > 0:
                text = self.ocr_detector.predict(cropped)
                fields.append(text)

        print(f"[INFO] OCR time: {time.time() - t_ocr:.2f}s")

        # Xác định loại CCCD
        version = "unknown"
        if len(fields) >= 7:
            version = "old"
            field_names = ["idNumber", "fullName", "dob", "gender", "nationality", "birthplace", "address1", "address2"]
        elif len(fields) == 5:
            version = "new"
            field_names = ["fullName", "idNumber", "gender", "dob", "nationality"]
        else:
            return {"success": False, "error": f"Không thể xác định phiên bản CCCD. Số field: {fields}"}

        result = {k: v for k, v in zip(field_names, fields)}
        result["success"] = True
        result["version"] = version
        return result

    def process_storage(self, video_file, front_image, back_image):
        front_path = self._save_temp_file(front_image, "front.jpg")
        back_path = self._save_temp_file(back_image, "_back.jpg")
        video_path = self._save_temp_file(video_file, "_video.mp4")

        url_front = self.s3_client.upload_file("cccd_front.jpg", front_path)
        url_back = self.s3_client.upload_file("cccd_back.jpg", back_path)
        url_video = self.s3_client.upload_file("video.mp4", video_path)

        log_result = {
            "url_cccd_front": url_front,
            "url_cccd_back": url_back,
            "url_video": url_video,
            "rate_success": True,
            "ocr_result": True,
            "error_message": "SUCCESS",
            "log": True
        }
        self.user_repository.save_verification(log_result)

        return {"success": True, "status": log_result}

    @staticmethod
    def _save_temp_file(file_obj, filename):
        path = os.path.join("/tmp", filename)
        if hasattr(file_obj, "save"):
            file_obj.save(path)
        elif isinstance(file_obj, str) and os.path.exists(file_obj):
            shutil.copy(file_obj, path)
        else:
            raise TypeError(f"Unsupported file object type: {type(file_obj)}")
        return path

    @staticmethod
    def _calc_success(result_dict):
        total = len(result_dict)
        passed = sum(1 for v in result_dict.values() if v)
        return round(passed / total, 2)