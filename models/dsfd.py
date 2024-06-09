from dsfd import DSFD
import torch

class DSFDModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = DSFD(device=self.device)
        self.model.load_weights("path/to/dsfd/weights.pth")

    def detect(self, img):
        # Image should be a PIL image
        return self.model.detect_faces(img)
