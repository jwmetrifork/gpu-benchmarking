# How to Run the Detectron2 Benchmark

## Setup Virtual Environment
```bash
python3 -m venv detectron2_venv
source detectron2_venv/bin/activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
pip install --upgrade pip setuptools wheel  
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Download COCO Dataset
```bash
mkdir datasets && cd datasets
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip "*.zip"
cd ..
```

## Run Benchmark
```bash
python benchmark.py --gpus 0,1,2 --image_dir datasets/val2017 --num_images 100
```

