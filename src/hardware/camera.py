import cv2
import time
import numpy as np
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import CAMERA_INDEX, YELLOW_TIME, N_FRAMES


class Camera:
    def __init__(self, index: int = CAMERA_INDEX, images_dir: str = "images"):
        self.cap = cv2.VideoCapture(index)
        self.images_dir = Path(images_dir)
        self.image_index = 0

        self.image_paths = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]) if self.images_dir.exists() else []

        if not self.cap.isOpened():
            print(f"[AVISO] Não foi possível abrir câmera {index}. Usando pasta: {images_dir}")
            self.cap.release()
            self.cap = None

            if not self.image_paths:
                raise RuntimeError(f"Nenhuma imagem encontrada em: {images_dir}")
        else:
            print(f"[OK] Câmera {index} aberta")
            time.sleep(0.5)

    def capture_frames(self, n: int = N_FRAMES) -> list[np.ndarray]:
        if self.cap is None:
            path = self.image_paths[self.image_index]

            print(f"[IMAGEM] Usando: {path}")

            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Não foi possível carregar a imagem: {path}")

            self.image_index += 1

            if self.image_index >= len(self.image_paths):
                self.image_index = 0

            return [img.copy() for _ in range(n)]

        interval = YELLOW_TIME / (n + 1)
        frames = []

        for _ in range(n):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            time.sleep(interval)

        if not frames:
            raise RuntimeError("No frames captured")

        return frames

    def release(self):
        if self.cap is not None:
            self.cap.release()