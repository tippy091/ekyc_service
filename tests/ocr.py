import cv2
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np

# Load YOLOv8 model (custom)
model_path = "/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/new_content_back.pt"  # ⚠️ Thay bằng model của bạn
model = YOLO(model_path)

# Class labels mapping
class_names = {
    0: 'Date of expirty', 1: 'Date of issue', 2: 'Place', 3: 'Place of birth',
    4: 'address_1', 5: 'address_2', 6: 'bottom_left', 7: 'bottom_right',
    8: 'cdate_of_birth', 9: 'cdate_of_expiry', 10: 'cdate_of_issue',
    11: 'cplace_of_birth', 12: 'date_of_expiry', 13: 'date_of_issue',
    14: 'place', 15: 'place_of_birth', 16: 'top_left', 17: 'top_right'
}

# Load VietOCR
vietocr_config = Cfg.load_config_from_name('vgg_seq2seq')
vietocr_config['weights'] = "/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/seq2seq.pth"  # ⚠️ Thay bằng checkpoint của bạn
vietocr_config['device'] = 'cpu'
vietocr_config['cnn']['pretrained'] = False
ocr_model = Predictor(vietocr_config)

# Inference function
def ocr_cccd_back(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]

    extracted_data = {}

    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = class_names.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        text = ocr_model.predict(crop_pil)

        if label in extracted_data:
            if isinstance(extracted_data[label], list):
                extracted_data[label].append(text)
            else:
                extracted_data[label] = [extracted_data[label], text]
        else:
            extracted_data[label] = text

    return extracted_data

# === Example usage ===
if __name__ == "__main__":
    img_path = "/Users/happy091/Downloads/CCCD Mới Dataset/OCR_CCCD.v1i.yolov5pytorch/train/images/20021580_01JARERNB2WRRB85DQ1P0J2XQE379632_back_jpg.rf.2bf25f58f9b272e9afb0f99a90c3727b.jpg"  # ⚠️ Ảnh CCCD mặt sau
    result = ocr_cccd_back(img_path)

    print("\n📄 Extracted Fields:")
    for k, v in result.items():
        print(f"{k}: {v}")