# Qwen3-VL Captioning Tool (vLLM)
* (Qwen3-VL-8B-Instruct)

## 1. Environment Setup
- Create and activate the Conda environment
  ```bash
  conda create -n qwen3-vl python=3.10 -y
  conda activate qwen3-vl
  ```
- Install the vLLM package
  ```bash
  pip install -U vllm openai
  ```
## 2. Run the vLLM Server
- Script: qwen_serve.sh
- Run Server (Hardware requirement (expected): RTX 4090 (24GB) x 2)
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 ./qwen_serve.sh
  ```
## 3. Run Captioning
- Script: qwen_captioning.py
  ```bash
  python qwen_captioning.py
  ```
## 4. Output
- Outputs are saved as a JSONL file (1 line = 1 video).
- It reads the existing output and skips already-processed video_name entries to resume (resume).

## Notes
- qwen_captioning.py assumes the following structure:
  - /home/dataset/video_eval/L{1..5}/{short,medium,long}/*.mp4
- The server-side --allowed-local-media-path restricts local paths accessible via file://....
- Video sampling is controlled by --media-io-kwargs in qwen_serve.sh (e.g., num_frames, fps).
