import argparse
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from skimage.transform import resize
from scipy import ndimage
import random
from torchvision.transforms.functional import center_crop

class SmartphoneDegradation:
    def __init__(self, dyn_range=None, jpg_quality=40, downscale_factor=2, noise_strength=2, blur=True):
        self.dyn_range = dyn_range
        self.jpg_quality = jpg_quality
        self.downscale_factor = downscale_factor
        self.noise_strength = noise_strength
        self.blur = blur

    def smartphone_blur(self, img):
        sharpened = img.filter(ImageFilter.SHARPEN)

        # Custom blur kernel
        kernel = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 4, 2, 1, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]

        angle = random.random() * 90
        power = (0.7 + random.random() * 1.3)

        rotated = resize(ndimage.rotate(resize(np.array(kernel), [17, 17], mode='constant'), angle), [5, 5]) ** power

        custom_blur = ImageFilter.Kernel(
            size=(5, 5),
            kernel=np.reshape(rotated, [-1]),
            scale=np.sum(rotated),
            offset=0
        )

        return sharpened.filter(custom_blur)

    def __call__(self, img):
        # 1) Downscale → Upscale
        fac = 1 + random.random() * (self.downscale_factor - 1)
        small = img.resize((int(img.width / fac), int(img.height / fac)), Image.LANCZOS)
        img = small.resize((img.width, img.height), Image.LANCZOS)

        # 2) Apply smartphone blur
        if self.blur:
            img = self.smartphone_blur(img)

        # 3) JPEG Compression
        buf = BytesIO()
        quality = min(int(self.jpg_quality + int(random.random() * (99 - self.jpg_quality))), 95)
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        arr = np.array(img, dtype=np.float32)

        # 4) Add noise
        noise = np.random.normal(0, self.noise_strength * random.random(), arr.shape)
        arr += noise

        # 5) Color Fringing (red channel shift)
        r = np.roll(arr[:, :, 0], random.randint(0, 1), axis=random.randint(0, 1))
        g, b = arr[:, :, 1], arr[:, :, 2]
        arr = np.dstack([r, g, b])

        # 6) Reduce Dynamic Range
        arr = np.clip(arr, 0, 255) / 255.0
        if self.dyn_range is not None:
            arr **= (self.dyn_range + random.random()*0.6)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)
    
class CenterCropVariableSize:
    def __init__(self, resolution):
        self.aspect_ratios = [1., 3/4, 2/3, 3/5, 4/5, 9/16]
        self.resolution = resolution

    def __call__(self, img, seed=42):
        # Randomly select a crop size within the specified range
        random.seed(seed)
        ratio = random.choice(self.aspect_ratios)
        crop_size = [self.resolution, self.resolution]
        crop_size[random.choice([0, 1])] = int((crop_size[random.choice([0, 1])] * ratio + 7) // 8 * 8)
        return center_crop(img, crop_size)
