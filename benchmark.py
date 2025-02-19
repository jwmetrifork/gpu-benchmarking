import time
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.data import MetadataCatalog
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

def setup_model(device):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = device  # Assign model to specific GPU
    return DefaultPredictor(cfg)

def load_images(image_dir, num_images=100):
    images = []
    for file in os.listdir(image_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            images.append(img)
            if len(images) >= num_images:
                break
    return images

def benchmark_model(predictor, images, device):
    torch.cuda.synchronize()
    start_time = time.time()
    
    for img in tqdm(images, desc=f"Benchmarking {device}"):
        _ = predictor(img)
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_time = total_time / len(images)
    fps = 1 / avg_time
    max_cameras = fps / 5  # Compute max cameras at 5 FPS per camera
    return avg_time, fps, max_cameras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--image_dir", type=str, default="datasets/val2017", help="Path to COCO val2017 images")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to benchmark")
    args = parser.parse_args()
    
    gpu_ids = args.gpus.split(",")
    images = load_images(args.image_dir, args.num_images)
    
    results = {}
    for gpu_id in gpu_ids:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Running benchmark on {device}")
        model = setup_model(device)
        avg_time, fps, max_cameras = benchmark_model(model, images, device)
        results[device] = {"Avg Time per Image (s)": avg_time, "FPS": fps, "Max Cameras @5FPS": int(max_cameras)}
    
    print("\n==== Benchmark Results ====\n")
    for device, metrics in results.items():
        print(f"{device}: {metrics}")

if __name__ == "__main__":
    main()
