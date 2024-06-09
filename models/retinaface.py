from retinaface import RetinaFace
import torch

class RetinaFaceModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = RetinaFace(device=self.device)

    def detect(self, img):
        # Image should be a PIL image
        return self.model.predict(img)
