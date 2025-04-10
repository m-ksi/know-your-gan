import os
from PIL import Image
from typing import Dict, Union, Iterable


class Logger:
    def __init__(self, path: str, keys: Iterable[str] = ()):
        self.path = path
        os.makedirs(os.path.join(path, "images"), exist_ok=True)
        os.makedirs(os.path.join(path, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(path, "states"), exist_ok=True)
        self.keys = keys
        self.files = {}
        for key in keys:
            self.add_key(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def add_key(self, key: str):
        if key in self.files:
            return
        filepath = os.path.join(self.path, f"metrics/{key}.txt")
        self.files[key] = open(filepath, "a")

    def log_metric(self, value: Union[float, int], key: str, step: int):
        if key not in self.files:
            self.add_key(key)
        try:
            formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
            self.files[key].write(f"{step}, {formatted_value}\n")
            self.files[key].flush()
        except Exception as e:
            print(f"[Logger] Failed to log metric {key} at step {step}: {e}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int):
        for key, value in metrics.items():
            self.log_metric(value, key, step)

    def log_image(self, image: Image.Image, step: int):
        image_path = os.path.join(self.path, f"images/{step:08}.png")
        try:
            image.save(image_path)
        except Exception as e:
            print(f"[Logger] Failed to save image at step {step}: {e}")

    def finish(self):
        for file in self.files.values():
            file.close()
        self.files.clear()
