#-------------------------------------#
#       General Utilities
#-------------------------------------#
import os
import random

import numpy as np
import torch
from PIL import Image


def get_classes(classes_path):
    """Load class names from file"""
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    """Initialize worker for DataLoader

    Args:
        worker_id: Worker ID passed automatically by PyTorch DataLoader
        rank: Process rank for distributed training
        seed: Base seed for random number generation
    """
    worker_seed = worker_id + rank + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cvtColor(image):
    """Convert image to RGB"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    image = image.convert('RGB')
    return image


def preprocess_input(image):
    """Preprocess image: normalize to 0-1"""
    image /= 255.0
    return image


def resize_image(image, size, letterbox_image):
    """Resize image with optional letterbox"""
    iw, ih = image.size
    w, h = size

    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)

    return new_image


def show_config(**kwargs):
    """Print configuration"""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(phi, save_dir="model_data"):
    """Download pretrained weights for YOLOv12"""
    import ssl
    import urllib.request

    urls = {
        'n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt',
        's': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt',
        'm': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt',
        'l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt',
        'x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt',
    }

    if phi not in urls:
        print(f"No pretrained weights available for yolo12{phi}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    filename = f"yolo12{phi}.pt"
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"Pretrained weights already exists: {save_path}")
        return save_path

    url = urls[phi]
    print(f"Downloading pretrained weights from {url}")
    try:
        # 创建不验证SSL的上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # 使用urllib下载
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())
        print(f"Downloaded to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download pretrained weights: {e}")
        print(f"Please manually download from: {url}")
        print(f"And save to: {save_path}")
        return None
