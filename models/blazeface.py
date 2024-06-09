from blazeface import BlazeFace
import torch

class BlazeFaceModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = BlazeFace().to(self.device)
        self.model.load_weights("path/to/blazeface/weights.pth")

    def detect(self, img):
        # Image should be a PIL image
        return self.model.predict_on_image(img)
