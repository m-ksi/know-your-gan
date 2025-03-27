import os

class Logger:
    def __init__(self, path, keys=()):
        self.path = path
        os.makedirs(os.path.join(path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(path, 'metrics'), exist_ok=True)
        self.keys = keys
        self.files = {}
        for key in keys:
            self.add_key(key)

    def add_key(self, key):
        self.files[key] = open(os.path.join(self.path, f"metrics/{key}.txt"), 'w')

    def log_metric(self, value, key, step):
        if key not in self.files:
            self.add_key(key)
        self.files[key].write(f"{step}, {value}\n")

    def log_metrics(self, d, step):
        for key, value in d.items():
            self.log_metric(value, key, step)

    def log_image(self, image, step):
        image.save(os.path.join(self.path, f'images/{step}.png'))

    def finish(self):
        for file in self.files:
            file.close()
