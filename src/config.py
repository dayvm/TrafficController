import os


def env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in ("1", "true", "yes", "sim", "s")

USE_CAMERA = env_bool("USE_CAMERA", False)

BASE_GREEN   = 30
STEP         = 10
MIN_GREEN    = 10
YELLOW_TIME  = 10

CAMERA_INDEX = 1
BAUD_RATE    = 9600
N_FRAMES     = 5

INPUT_SIZE   = 224
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
NUM_CLASSES  = 9

CLASS_MAP = {
    0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1),
    4: (2, 0), 5: (0, 2), 6: (2, 1), 7: (1, 2), 8: (2, 2),
}