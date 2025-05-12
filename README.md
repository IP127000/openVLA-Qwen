**Read this in other languages: [English](README.md), [中文](README_zh.md).**

# OpenVLA Lightweight Version

Lightweight version of OpenVLA, based on the llava-next framework

Features:

1.The main model is only 0.5B, using qwen2-0.5B.

2.Does not use RLDS format datasets, instead fine-tuned using mllm format.

3.Does not occupy inherent tokens of the LLM model, instead using an additional 271 tokens to represent actions.
# How to use

## Installation

Follow the steps below to install the OpenVLA-Qwen lightweight version.

### 1. Clone the repository

Clone the repository to your local machine:

```bash
git clone https://github.com/IP127000/openVLA-Qwen.git
cd openVLA-Qwen
```
### 2. Create a new conda environment and activate it:
```bash
conda create -n vla-qwen python=3.10 -y
conda activate vla-qwen
```
### 3. Install the required dependencies
```bash
pip install --upgrade pip
pip install -e ".[train]"
```
### 4. Download the weights from huggingface
```bash
cd openVLA-Qwen/models
git clone git clone https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov
```
### 5. Change the file path in scripts/train/finetune_ov_vla.sh
```bash
PREV_STAGE_CHECKPOINT="models/llava-onevision-qwen2-0.5b-ov" 
--data_path scripts/train/vla.yaml
--image_folder /images \
```
### 6. Change the data paht in scripts/train/vla.yaml
```bash
json_path: /root/data/vla_llava.json
```

### 7. start to train
```bash
cd /openVLA-Qwen2-0.5B
./scripts/train/finetune_ov_vla.sh
```
## Acknowledgements

This project makes use of the following open-source projects, and I would like to express my gratitude to their creators:

- [OpenVLA](https://github.com/openvla/openvla) - For providing the core structure and functionality that inspired OpenVLA-Qwen.
- [LLaVA_OpenVLA](https://github.com/Darren-greenhand/LLaVA-Next) - For his contribution to data preprocessing techniques.
