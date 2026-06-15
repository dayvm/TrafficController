import os
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from hardware.serial_comm import ArduinoSerial
from timing import next_times
from vision.classifier import Classifier


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(PROJECT_DIR / "model" / "EFFICIENTNET_best.pth")
)

ARDUINO_PORT = os.getenv("ARDUINO_PORT") or None
SIMULATE_ARDUINO = os.getenv("SIMULATE_ARDUINO", "0").lower() in (
    "1",
    "true",
    "yes",
    "sim",
    "s",
)

app = FastAPI(title="TrafficController API")

classifier: Optional[Classifier] = None
arduino: Optional[ArduinoSerial] = None

classifier_lock = threading.Lock()
arduino_lock = threading.Lock()


def get_classifier() -> Classifier:
    global classifier

    with classifier_lock:
        if classifier is None:
            classifier = Classifier(MODEL_PATH)

    return classifier


def get_arduino() -> ArduinoSerial:
    global arduino

    with arduino_lock:
        if arduino is None:
            arduino = ArduinoSerial(port=ARDUINO_PORT, simulate=SIMULATE_ARDUINO)

    return arduino


def upload_to_bgr_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(image)
    return rgb[:, :, ::-1]


def send_times_to_arduino(right_green: int, bottom_green: int) -> tuple[bool, Optional[str]]:
    try:
        board = get_arduino()

        if board.simulate:
            board.send("T", f"{right_green},{bottom_green}")
            return True, None

        with arduino_lock:
            msg = f"T,{right_green},{bottom_green}\n"
            board.ser.write(msg.encode())

        return True, None

    except Exception as exc:
        return False, str(exc)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = PROJECT_DIR / "web" / "index.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html nao encontrado")

    return html_path.read_text(encoding="utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem valida")

    file_bytes = await file.read()

    try:
        frame_bgr = upload_to_bgr_image(file_bytes)

        clf = get_classifier()
        class_id, right_cars, bottom_cars = clf.predict(frame_bgr)

        right_green, bottom_green = next_times(right_cars, bottom_cars)
        arduino_sent, arduino_error = send_times_to_arduino(right_green, bottom_green)

        return {
            "class_id": class_id,
            "right_cars": right_cars,
            "bottom_cars": bottom_cars,
            "right_green": right_green,
            "bottom_green": bottom_green,
            "arduino_sent": arduino_sent,
            "arduino_error": arduino_error,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.on_event("shutdown")
def shutdown():
    global arduino

    if arduino is not None:
        arduino.close()
        arduino = None