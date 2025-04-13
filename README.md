# OpenVLA Lightweight Version

OpenVLA的轻量化版本，使用

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

We would like to express our gratitude to the following repositories:

- [OpenVLA](https://github.com/openvla/openvla) - for providing the core structure and functionality that inspired OpenVLA-Qwen.
- [LLaVA_OpenVLA](https://github.com/Darren-greenhand/LLaVA_OpenVLA) - for their contribution to data preprocessing techniques.
