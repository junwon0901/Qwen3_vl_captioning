## Experiment setup: Qwen3-VL-30B-A3B-Thinking

This section describes how to serve **Qwen3-VL-30B-A3B-Thinking** with **vLLM** and run batch video captioning via the OpenAI-compatible endpoint.

```bash
# 1) Create and activate environment
conda create -n qwen3-vl python=3.10 -y
conda activate qwen3-vl

# 2) Install vLLM
pip install -U vllm

# 3) Start vLLM server
# Script: qwen_serve.sh
# Hardware requirement: RTX 4090 (24GB) × 4 (expected)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./qwen_serve.sh

# 4) Run batch captioning
# Script: qwen3_captioning.py
python qwen3_captioning.py
