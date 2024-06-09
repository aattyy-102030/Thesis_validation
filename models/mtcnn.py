from facenet_pytorch import MTCNN
import torch

class MTCNNModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True, device=self.device)

    def detect(self, img):
        # Image should be in PIL format
        boxes, _ = self.model.detect(img)
        return boxes
