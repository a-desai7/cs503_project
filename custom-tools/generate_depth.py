from transformers import pipeline
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import cv2

def process_image(img_path, out_path, pipe):
    image = Image.open(img_path).convert('RGB')
    depth = pipe(image)["depth"]
    # Convert PIL Image to numpy array
    depth_np = np.array(depth)
    # Normalize to 0-255 for saving as PNG
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    if depth_max - depth_min > 1e-6:
        depth_np = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_np = np.zeros_like(depth_np)
    depth_np = (depth_np * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, depth_np)

def generate_depths(img_dir, depth_dir, exts=(".png", ".jpg", ".jpeg"), use_alt_model=False):
    model_name = "xingyang1/Distill-Any-Depth-Large-hf"
    if use_alt_model:
        model_name = "Intel/dpt-hybrid-midas"  # Example: a lower-quality model
    pipe = pipeline(task="depth-estimation", model=model_name)
    for root, _, files in os.walk(img_dir):
        for fname in tqdm(files):
            if not fname.lower().endswith(exts):
                continue
            img_path = os.path.join(root, fname)
            rel_path = os.path.relpath(img_path, img_dir)
            out_path = os.path.join(depth_dir, rel_path)
            process_image(img_path, out_path, pipe)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate depth maps using HuggingFace Transformers depth-estimation pipeline.")
    parser.add_argument('--img_dir', type=str, required=True, help='Input image directory or parent directory')
    parser.add_argument('--depth_dir', type=str, required=False, help='Output depth directory (ignored in batch mode)')
    parser.add_argument('--batch_mode', action='store_true', help='If set, process all subdirectories in img_dir, creating _depth folders for each')
    parser.add_argument('--alt_model', action='store_true', help='Use alternative (lower quality) depth model')
    args = parser.parse_args()

    if args.batch_mode:
        parent_dir = args.img_dir
        for entry in os.listdir(parent_dir):
            entry_path = os.path.join(parent_dir, entry)
            if os.path.isdir(entry_path):
                out_dir = entry_path + '_depth'
                print(f"Processing {entry_path} -> {out_dir}")
                generate_depths(entry_path, out_dir, use_alt_model=args.alt_model)
    else:
        if args.depth_dir is None:
            raise ValueError("--depth_dir must be specified unless --batch_mode is used.")
        generate_depths(args.img_dir, args.depth_dir, use_alt_model=args.alt_model)
