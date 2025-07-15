import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN
import torch
from PIL import Image

def padding_face(box: np.ndarray, padding=None):
    """
    Pad the given bounding box.
    Parameters:
        box (np.ndarray): A bounding box in the format [x1, y1, x2, y2].
        padding (float or int, optional): Padding value. If a float is provided, it's a scaling factor. If an int is provided, it's added to the width and height.

    Returns:
        np.ndarray: Padded bounding box.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    if padding:
        if isinstance(padding, float):
            w = w * padding
            h = h * padding
        else:
            w = w + padding
            h = h + padding

    x1 = cx - w // 2
    x2 = cx + w // 2
    y1 = cy - h // 2
    y2 = cy + h // 2

    box = np.clip([x1, y1, x2, y2], 0, np.inf).astype(np.uint32)
    return box


def extract_face(img: np.ndarray, model: MTCNN, padding=None, min_prob=0.9):
    boxes, prob, landmarks = model.detect(img, landmarks=True)

    if boxes is not None:
        boxes = boxes[prob > min_prob]

        max_area = 0
        max_box = [0, 0, 0, 0]
        max_landmarks = []

        for i, box in enumerate(boxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) > max_area:
                max_box = padding_face(box, padding)
                max_area = (x2 - x1) * (y2 - y1)
                max_landmarks = landmarks[i]

        x1, y1, x2, y2 = max_box
        face = img[y1:y2, x1:x2, ...]
        return face, max_box, max_landmarks

    return None, None, None

def normalize_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(lab)
    l = cv.equalizeHist(l)
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)

def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face so that eyes are horizontal.
    landmarks: array of 5 facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    """
    if landmarks is None or len(landmarks) < 2:
        return img  # fallback: no alignment

    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])

    # Tính góc cần xoay
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Tính trung điểm giữa 2 mắt
    center_x = float((left_eye[0] + right_eye[0]) / 2)
    center_y = float((left_eye[1] + right_eye[1]) / 2)
    center = (center_x, center_y)

    # Tạo ma trận xoay và xoay ảnh
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return aligned

def face_transform(face: np.ndarray, model_name="base", device="cpu"):
    """
    Preprocesses a face image for deep learning models.

    Args:
        face (numpy.ndarray): The input face image as a NumPy array.
        size (int or tuple): The desired size for the preprocessed image.
        model_name (str): The name of the model for which preprocessing is done.
        device (str or torch.device): The device to perform preprocessing on.

    Returns:
        torch.Tensor: The preprocessed face image as a PyTorch tensor.
    """

    if isinstance(device, str):
        if (device == "cuda" or device == "gpu") and torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device("cpu")

    if model_name == "base":
        mean = (127.5, 127.5, 127.5)
        std = 1
        size = (64, 64)
    elif model_name == "VGG-Face2":
        mean = (127.5, 127.5, 127.5)
        size = (160, 160)
        std = 128

    face = cv.resize(face, size)

    face = (face.astype(np.float32) - mean) / std

    face = torch.from_numpy(face)

    face = face.permute(2, 0, 1)

    if len(face.shape) == 3:
        face = face[None, ...]
    face = face.to(device)
    return face.float()


def get_image(filename: str) -> np.ndarray:
    """Load an image from the specified filename using OpenCV's imread function"""
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img
