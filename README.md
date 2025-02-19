# GPU Benchmarking with Detectron2

This repository provides a benchmarking tool for evaluating the performance of GPUs using the Detectron2 framework. The benchmark measures the average time per image, frames per second (FPS), and the maximum number of cameras that can be processed at 5 FPS.

## Setup

### 1. Create a Virtual Environment
```bash
python3 -m venv detectron2_venv
source detectron2_venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install --upgrade pip setuptools wheel  
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Download COCO Dataset
```bash
mkdir datasets && cd datasets
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip "*.zip"
cd ..
```

## Running the Benchmark

To run the benchmark, use the following command:
```bash
python benchmark.py --gpus 0,1,2 --image_dir datasets/val2017 --num_images 100
```
- `--gpus`: Comma-separated list of GPU IDs to benchmark.
- `--image_dir`: Path to the directory containing COCO validation images.
- `--num_images`: Number of images to use for benchmarking.

## Benchmark Results

The benchmark script will output the average time per image, FPS, and the maximum number of cameras that can be processed at 5 FPS for each GPU.

## Files

- `requirements.txt`: List of Python dependencies.
- `howtorun.md`: Detailed instructions on how to set up and run the benchmark.
- `benchmark.py`: The main benchmarking script.
