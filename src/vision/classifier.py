import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import Counter
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import CLASS_MAP, MEAN, STD
from vision.model import EfficientNet

class Classifier:
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)

        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("model_state")
            or ckpt
        )

        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, frame_bgr: np.ndarray) -> tuple[int, int, int]:
        img = Image.fromarray(frame_bgr[:, :, ::-1])
        x = self._transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cls = int(self.model(x).argmax(1).item())
        right, bottom = CLASS_MAP[cls]
        return cls, right, bottom

    def predict_majority(self, frames: list[np.ndarray]) -> tuple[int, int, int]:
        votes = [self.predict(f)[0] for f in frames]
        cls = Counter(votes).most_common(1)[0][0]
        right, bottom = CLASS_MAP[cls]
        print(f"  votes={votes}  winner={cls}")
        return cls, right, bottom