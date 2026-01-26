# Qwen3-VL Captioning Tool (vLLM)

## 1. 환경 설정
- Conda 환경 생성 및 활성화
- vLLM 패키지 설치

## 2. vLLM 서버 실행 (Qwen3-VL-30B-A3B-Thinking)
- Script: `qwen_serve.sh`
- 서버 실행: `CUDA_VISIBLE_DEVICES=0,1,2,3 ./qwen_serve.sh`
- Hardware requirement (expected): RTX 4090 (24GB) × 4

## 3. 데이터 준비
- `qwen3_captioning.py`는 아래 구조를 가정합니다:
  - `/home/dataset/video_eval/L{1..5}/{short,medium,long}/*.mp4`

## 4. Captioning 실행
- Script: `qwen3_captioning.py`

## 5. 출력
- 출력은 JSONL 파일로 저장됩니다(1 line = 1 video).
- 기존 output을 읽어 이미 처리된 `video_name`은 스킵하여 재개(resume)합니다.

## Notes
- 서버 측 `--allowed-local-media-path`로 `file://...` 접근 가능한 로컬 경로가 제한됩니다.
- 비디오 샘플링은 `qwen_serve.sh`의 `--media-io-kwargs` (예: `num_frames`, `fps`)에서 제어합니다.


### 1) 환경 설정
```bash
conda create -n qwen3-vl python=3.10 -y
conda activate qwen3-vl
```

```bash
pip install -U vllm openai
```

### 2) vLLM 서버 실행 (Qwen3-VL-30B-A3B-Thinking)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./qwen_serve.sh
```

### 3) Captioning 실행
```bash
python qwen3_captioning.py
```