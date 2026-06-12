import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../TrafficController/src"))

from config import BASE_GREEN, N_FRAMES
from vision.classifier import Classifier
from hardware.camera import Camera
from hardware.serial_comm import ArduinoSerial
from timing import next_times

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",     default=None)
    p.add_argument("--camera",   type=int, default=1)
    p.add_argument("--model",    default=os.path.join(os.path.dirname(__file__), "../TrafficController/model/EFFICIENTNET_best.pth"))
    p.add_argument("--n-frames", type=int, default=N_FRAMES)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--images-dir", default="images")
    return p.parse_args()

def classify_yellow(cam: Camera, clf: Classifier, n: int) -> tuple[int, int]:
    frames = cam.capture_frames(n=n)
    cls, right, bottom = clf.predict_majority(frames)
    print(f"  class={cls}  right={right}  bottom={bottom}")
    return right, bottom

def main():
    args = parse_args()
    clf    = Classifier(args.model)
    cam = Camera(index=args.camera, images_dir=args.images_dir)
    serial = ArduinoSerial(port=args.port, simulate=args.simulate)
    t_right = t_bottom = BASE_GREEN
    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n[cycle {cycle}]")
            print(f"RIGHT  GREEN {t_right}s")
            serial.send("R", t_right)
            serial.wait_done()
            r, b = classify_yellow(cam, clf, args.n_frames)
            t_right_next, t_bottom_next = next_times(r, b)
            print(f"BOTTOM GREEN {t_bottom}s")
            serial.send("B", t_bottom)
            serial.wait_done()
            r, b = classify_yellow(cam, clf, args.n_frames)
            t_right, t_bottom = next_times(r, b)
            t_right  = t_right_next
            t_bottom = t_bottom_next
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        serial.close()

if __name__ == "__main__":
    main()