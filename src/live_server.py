"""Serve sensor capture and local window inference for the DASIG model."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import json
from pathlib import Path
import re
import socket
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_checkpoint  # noqa: E402

PHONE_DIR = ROOT / "data" / "phone_records"
PHONE_LOCK = threading.Lock()


class LiveHandler(BaseHTTPRequestHandler):
    model = None
    checkpoint = None
    page = b""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", self.page)
        if self.path == "/api/status":
            return self._json(200, {"ready": True, "classes": self.checkpoint["class_names"],
                                   "window_samples": self.checkpoint["window_samples"],
                                   "window_seconds": self.checkpoint["window_seconds"],
                                   "sample_rate_hz": self.checkpoint["sample_rate_hz"]})
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/api/phone/record":
            return self._record_phone_rows()
        if self.path != "/api/predict":
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100_000:
                raise ValueError("Request body must be between 1 byte and 100 KB.")
            payload = json.loads(self.rfile.read(length))
            window = np.asarray(payload.get("window"), dtype=np.float32)
            expected = (int(self.checkpoint["window_samples"]), len(self.checkpoint["channels"]))
            if window.shape != expected or not np.isfinite(window).all():
                raise ValueError(f"Expected {expected[0]} finite readings with {expected[1]} sensor channels.")
            mean = self.checkpoint["mean"].numpy()
            scale = self.checkpoint["scale"].numpy()
            normalized = ((window - mean) / scale).astype(np.float32)
            import torch

            with torch.inference_mode():
                logits = self.model(torch.from_numpy(normalized[None, ...]))
                probs = torch.softmax(logits, dim=1)[0].numpy()
            order = np.argsort(probs)[::-1][:3]
            classes = self.checkpoint["class_names"]
            return self._json(200, {
                "label": classes[int(order[0])],
                "confidence": float(probs[order[0]]),
                "top": [{"label": classes[int(i)], "probability": float(probs[i])} for i in order],
            })
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})

    def _record_phone_rows(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 250_000:
                raise ValueError("Request body must be between 1 byte and 250 KB.")
            payload = json.loads(self.rfile.read(length))
            session_id = str(payload.get("session_id", ""))
            rows = payload.get("rows")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
                raise ValueError("Invalid session id.")
            if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
                raise ValueError("A batch must contain between 1 and 500 rows.")
            cleaned = []
            for row in rows:
                if not isinstance(row, list) or len(row) != 8:
                    raise ValueError("Each row must be [time_s, label, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].")
                timestamp, label, *values = row
                if label not in ("standard", "abrupt", "unknown"):
                    raise ValueError("Labels must be standard, abrupt, or unknown.")
                numeric = [float(timestamp), *(float(value) for value in values)]
                if not np.isfinite(numeric).all():
                    raise ValueError("Sensor data must be finite numbers.")
                cleaned.append([numeric[0], label, *numeric[1:]])
            PHONE_DIR.mkdir(parents=True, exist_ok=True)
            path = PHONE_DIR / f"{session_id}.csv"
            with PHONE_LOCK, path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if stream.tell() == 0:
                    writer.writerow(["time_s", "label", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
                writer.writerows(cleaned)
            self._json(200, {"saved": len(cleaned), "file": str(path.relative_to(ROOT))})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})

    def _json(self, status, payload):
        self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def _send(self, status, content_type, content):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            super().log_message(fmt, *args)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=str(ROOT / "models" / "dasig_forearm_cnn_lstm.pt"))
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.model)
    LiveHandler.model = model
    LiveHandler.checkpoint = checkpoint
    LiveHandler.page = (ROOT / "app" / "phone.html").read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), LiveHandler)
    server.daemon_threads = True
    print("Live phone classifier and recording receiver are ready.")
    print(f"On this computer: http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("192.0.2.1", 1))
                lan_ip = sock.getsockname()[0]
            phone_url = f"http://{lan_ip}:{args.port}"
            print(f"On a phone on the same Wi-Fi: {phone_url}")
            try:
                import qrcode
                qr = qrcode.QRCode(border=2)
                qr.add_data(phone_url)
                qr.make(fit=True)
                print("\nScan this QR code with your phone camera:")
                qr.print_ascii(invert=True)
                qr_path = ROOT / "data" / "phone_qr.png"
                qr_path.parent.mkdir(parents=True, exist_ok=True)
                qr.make_image(fill_color="black", back_color="white").save(qr_path)
            except ImportError:
                pass
        except OSError:
            pass
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live classifier…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
