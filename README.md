**Read this in other languages: [English](README.md), [中文](README_zh.md).**
# OpenVLA Lightweight Version

OpenVLA的轻量化版本，基于llava-next框架
特点：
1.主体模型仅0.5B，使用qwen2-0.5B
2.不使用RLDS格式数据集，使用mllm格式进行微调
3.不占用LLM模型固有tokens,使用新增的271个tokens表示动作

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
## Acknowledgements

I would like to express my gratitude to the following repositories:

- [OpenVLA](https://github.com/openvla/openvla) - For providing the core structure and functionality that inspired OpenVLA-Qwen.
- [LLaVA_OpenVLA](https://github.com/Darren-greenhand/LLaVA-Next) - For his contribution to data preprocessing techniques.
