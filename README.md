# Qwen3-VL Captioning Tool

Qwen3-VL을 **vLLM(OpenAI-compatible endpoint)** 로 서빙한 뒤, 로컬 비디오 디렉터리를 순회하며 캡션을 생성하는 스크립트입니다.

---

## 1. 환경 설정

### 1) Conda 환경 생성 및 활성화

```bash
conda create -n qwen3-vl python=3.10 -y
conda activate qwen3-vl
2) vLLM 설치
bash
코드 복사
pip install -U vllm
2. vLLM 서버 실행 (Qwen3-VL-30B-A3B-Thinking)
1) 서버 실행 스크립트 준비
Script: qwen_serve.sh

bash
코드 복사
chmod +x qwen_serve.sh
2) 서버 실행
bash
코드 복사
CUDA_VISIBLE_DEVICES=0,1,2,3 ./qwen_serve.sh
Hardware requirement (expected): RTX 4090 (24GB) × 4

3. Captioning 실행

Script: qwen3_captioning.py

python qwen3_captioning.py

4. 출력 (JSONL)

출력은 JSONL이며, 한 줄에 한 비디오 결과가 저장됩니다.

스크립트는 기존 output을 읽어 이미 처리된 video_name은 스킵(resume) 합니다.

예시(개념):

{"video_name":"L1/short/xxx.mp4","caption":"..."}