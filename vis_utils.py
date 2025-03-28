import os
from PIL import Image

def save_gif(path):
    image_names = os.listdir(path)
    images = [Image.open(os.path.join(path, n)) for n in image_names]
    images[0].save(os.path.join(path, 'output.gif'),
                   save_all=True,
                   append_images=images[1:],
                   duration=200,
                   loop=0)
