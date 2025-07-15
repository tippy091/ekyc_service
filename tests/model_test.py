from ultralytics import YOLO
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path="/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/content.pt", force_reload=True)

# In ra tên các class mà mô hình detect được
print(model.names)  # Ví dụ: {0: 'top_left', 1: 'top_right', ...}

print(type(model))

from ultralytics import YOLO

# Load model
model = YOLO("/Volumes/Dev/HDC/python-ekyc-service/ekyc/models/weights/new_content_back.pt")  # Thay bằng đường dẫn model của bạn

# In ra lớp (class) mô hình nhận diện được
print(model.names)        # Trả về dict, ví dụ: {0: 'top_left', 1: 'top_right', ...}
print(type(model.names))  #