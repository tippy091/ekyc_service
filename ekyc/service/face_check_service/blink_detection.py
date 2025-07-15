import numpy as np
from imutils import face_utils
import os
import cv2 as cv
import torch
import dlib
import math
from time import time



class BlinkDetector():

    '''Detecting eye blinking in facial images'''

    def __init__(self):
        landmark_path = os.path.join(os.path.dirname(__file__), 'landmarks/shape_predictor_68_face_landmarks.dat')

        self.predictor_eyes = dlib.shape_predictor(landmark_path)

        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 2
        self.counter = 0
        self.total = 0

        # Count blink overtime
        self.blink_start_time = None
        self.last_ear_below_thresh = False
        self.min_blink_duration = 0.1

    
    def eye_blink(self, rgb_image: np.array, rect, thresh=1):
        if isinstance(rect, torch.Tensor):
            rect = dlib.rectangle(*rect.long())
        elif isinstance(rect, (np.ndarray, list, tuple)):
            rect = np.array(rect).astype(np.uint32)
            rect = dlib.rectangle(*rect)

        gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        shape = self.predictor_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < self.EYE_AR_THRESH:
            if not self.last_ear_below_thresh:
                self.blink_start_time = time()
                self.last_ear_below_thresh = True
        else:
            if self.last_ear_below_thresh:
                blink_duration = time() - self.blink_start_time
                if blink_duration >= self.min_blink_duration:
                    self.total += 1
                else:
                    self.last_ear_below_thresh = False

        if self.total >= thresh:
            self.total = 0
            return True
        return False

    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = math.dist(eye[1], eye[5])
        B = math.dist(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = math.dist(eye[0], eye[3])  
        # Compute the eye aspect ratio

        ear = (A + B) / (2.0 * C)

        return ear

if __name__ == '__main__':
    blink_detector = BlinkDetector()
              
        
        

        