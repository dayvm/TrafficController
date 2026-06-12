import serial
import serial.tools.list_ports
import time
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BAUD_RATE


class ArduinoSerial:
    def __init__(self, port: str = None, simulate: bool = False):
        self.simulate = simulate
        self.ser = None

        if self.simulate:
            print("[SIMULADO] Arduino desativado")
            return

        if port is None:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            if not ports:
                raise RuntimeError("No serial port found")
            port = ports[0]

        self.ser = serial.Serial(port, BAUD_RATE, timeout=5)
        time.sleep(2.0)

    def send(self, lane: str, duration: int):
        msg = f"{lane},{duration}\n"

        if self.simulate:
            print(f"[SIMULADO] Enviaria para Arduino: {msg.strip()}")
            return

        self.ser.write(msg.encode())

    def wait_done(self):
        if self.simulate:
            print("[SIMULADO] DONE")
            return

        while True:
            line = self.ser.readline().decode(errors="ignore").strip()

            if line == "DONE":
                return

            if line == "":
                raise TimeoutError("Arduino não respondeu DONE")

    def close(self):
        if self.ser:
            self.ser.close()