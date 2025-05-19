import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Load MiDaS model from torch.hub
midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
midas.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas.to(device)

# MiDaS transforms
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midas_transforms.dpt_transform

def process_image(img_path, out_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    # Normalize to 0-255 for saving as PNG
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)
    depth = (depth * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, depth)

def generate_depths(img_dir, depth_dir, exts=(".png", ".jpg", ".jpeg")):
    for root, _, files in os.walk(img_dir):
        for fname in tqdm(files):
            if not fname.lower().endswith(exts):
                continue
            img_path = os.path.join(root, fname)
            rel_path = os.path.relpath(img_path, img_dir)
            out_path = os.path.join(depth_dir, rel_path)
            process_image(img_path, out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate depth maps using MiDaS.")
    parser.add_argument('--img_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--depth_dir', type=str, required=True, help='Output depth directory')
    args = parser.parse_args()
    generate_depths(args.img_dir, args.depth_dir)
