import cv2
import torch
from facenet_pytorch import MTCNN
from blazeface import BlazeFace
from dsfd import DSFD
from retinaface import RetinaFace
import numpy as np

class FaceDetection:
    def __init__(self):
        # Initialize all models
        self.mtcnn = MTCNN(keep_all=True)
        self.blazeface = BlazeFace()
        self.dsfd = DSFD()
        self.retinaface = RetinaFace()

    def detect_faces_mtcnn(self, image):
        boxes, _ = self.mtcnn.detect(image)
        return boxes

    def detect_faces_blazeface(self, image):
        boxes = self.blazeface.predict_on_image(image)
        return boxes

    def detect_faces_dsfd(self, image):
        boxes = self.dsfd.detect_faces(image)
        return boxes

    def detect_faces_retinaface(self, image):
        boxes = self.retinaface.predict(image)
        return boxes

    def detect_faces(self, image, model_name):
        if model_name == 'mtcnn':
            return self.detect_faces_mtcnn(image)
        elif model_name == 'blazeface':
            return self.detect_faces_blazeface(image)
        elif model_name == 'dsfd':
            return self.detect_faces_dsfd(image)
        elif model_name == 'retinaface':
            return self.detect_faces_retinaface(image)
        else:
            raise ValueError("Unknown model name")

if __name__ == "__main__":
    image_path = 'data/images/sample.jpg'
    image = cv2.imread(image_path)
    detector = FaceDetection()
    
    models = ['mtcnn', 'blazeface', 'dsfd', 'retinaface']
    for model in models:
        boxes = detector.detect_faces(image, model)
        print(f"{model} detected boxes: {boxes}")
