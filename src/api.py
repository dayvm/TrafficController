import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from vision.classifier import Classifier


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(PROJECT_DIR / "model" / "EFFICIENTNET_best.pth")
)

app = FastAPI(title="TrafficController API")

classifier: Optional[Classifier] = None


def get_classifier() -> Classifier:
    """
    Carrega o modelo apenas quando chegar a primeira imagem.
    Depois disso, reutiliza o mesmo modelo nas próximas requisições.
    """
    global classifier

    if classifier is None:
        classifier = Classifier(MODEL_PATH)

    return classifier


def upload_to_bgr_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(image)

    # O classificador atual espera imagem em BGR, como o OpenCV entrega.
    bgr = rgb[:, :, ::-1]

    return bgr


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = PROJECT_DIR / "web" / "index.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html não encontrado")

    return html_path.read_text(encoding="utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem válido")

    file_bytes = await file.read()

    try:
        frame_bgr = upload_to_bgr_image(file_bytes)
        clf = get_classifier()
        class_id, right_cars, bottom_cars = clf.predict(frame_bgr)

        return {
            "class_id": class_id,
            "right_cars": right_cars,
            "bottom_cars": bottom_cars,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))