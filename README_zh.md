**Read this in other languages: [English](README.md), [中文](README_zh.md).**

# OpenVLA Lightweight Version

基于llava-next框架的OpenVLA轻量版。

特性：

主模型只有0.5B，使用qwen2-0.5B。

不使用RLDS格式的数据集，而是采用mllm格式进行微调。

不占用LLM模型的固有token，而是使用额外的271个token来表示动作。
# 如何使用

## Installation

按照以下步骤安装OpenVLA-Qwen轻量版。

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
### 7. 开始训练
```bash
cd /openVLA-Qwen2-0.5B
./scripts/train/finetune_ov_vla.sh
```
## 致谢

本项目使用了以下开源项目，在此向其开发者表示感谢：

- [OpenVLA](https://github.com/openvla/openvla) - 感谢提供核心结构和功能，启发了OpenVLA-Qwen。
- [LLaVA_OpenVLA](https://github.com/Darren-greenhand/LLaVA-Next) - 感谢他对数据处理技术的贡献。
