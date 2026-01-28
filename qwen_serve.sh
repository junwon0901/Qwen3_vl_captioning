#!/usr/bin/env bash

set -euo pipefail

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 1 \
  --max-model-len 10000 \
  --gpu-memory-utilization 0.80 \
  --allowed-local-media-path /home/dataset/video_eval \
  --media-io-kwargs '{"video":{"num_frames":2048,"fps":2}}' \
  --seed 42
