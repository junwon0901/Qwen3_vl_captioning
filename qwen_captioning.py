import time
import json
import random
import os
import io
import contextlib
from pathlib import Path

# Video URL compatibility depends on the backend library/version. You can override the
# default backend by setting FORCE_QWENVL_VIDEO_READER to torchvision, decord, or torchcodec
# before running this script.

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

base_dir = Path("/home/dataset/video_eval")
video_index_file = base_dir / "VE-500.json"

output_file = Path("captions_qwen3-32B-instruct.jsonl")
failed_file = Path("failed_video.json")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
device = next(iter(model.parameters())).device

processed_videos = set()
captions_by_name = {}
if output_file.exists():
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                video_name = data.get("video_name")
                caption = data.get("caption")
                if video_name and caption is not None:
                    processed_videos.add(video_name)
                    captions_by_name[video_name] = caption
            except json.JSONDecodeError:
                continue
    print(f"Found {len(processed_videos)} already processed videos. Skipping them...")

if not video_index_file.exists():
    raise FileNotFoundError(f"Video index file not found: {video_index_file}")

with open(video_index_file, "r", encoding="utf-8") as f:
    video_index = json.load(f)

video_entries = []
for group in video_index:
    for item in group.get("items", []):
        video_dir = Path(item["path"])
        video_path = video_dir / item["video_id"]
        video_entries.append(video_path)

print(f"Loaded {len(video_entries)} videos from {video_index_file}")

ordered_video_names = []
for video_path in video_entries:
    try:
        video_name = video_path.relative_to(base_dir).as_posix()
    except ValueError:
        video_name = video_path.as_posix()
    ordered_video_names.append(video_name)

prompt_options = [
    "Describe this video in detail, focusing on the spatio-temporal dynamics. Describe exactly how objects and agents move, change, and occupy space over time within the scene.",
    "Give a detailed account of everything shown in the video, capturing all visible specifics. Describe events in the exact order they appear over time. Ensure that you describe the sequence of events exactly as they occur, without skipping any steps.",
    "Describe the video in detail, paying special attention to how objects and people interact with each other. Capture the precise timing and nature of every contact, movement, and reaction shown in the footage.",
    "Thoroughly describe the video’s visual narrative, capturing every visible detail from start to end. Emphasize how actions unfold and how the scene transitions logically over time.",
    "Provide a thorough description of every detail, explicitly prioritizing spatio-temporal dynamics. Capture all visual elements, and focus on their spatial positions, movement trajectories, and how the scene evolves over time."
]

def log_failed(video_name, reason):
    record = {
        "video_name": video_name,
        "reason": reason,
    }
    with open(failed_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_captions_ordered():
    tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for name in ordered_video_names:
            caption = captions_by_name.get(name)
            if caption is None:
                continue
            record = {
                "video_name": name,
                "caption": caption,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(output_file)

def run_one_video(video_path: Path, video_name: str):
    selected_prompt_index = random.randrange(len(prompt_options))
    selected_prompt = prompt_options[selected_prompt_index]
    print(f"  Prompt selected: {selected_prompt_index + 1}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": 2,
                    "max_frames": 2048,
                },
                {"type": "text", "text": selected_prompt},
            ],
        }
    ]

    start = time.time()

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        image_inputs, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    if "video_reader_backend decord error" in err_buf.getvalue():
        print("  Decord failed; skipping this video.")
        log_failed(video_name, "decord_error")
        return False

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    elapsed_time = time.time() - start

    prompt_len = inputs["input_ids"].shape[1]
    gen_only_ids = generated_ids[:, prompt_len:]
    caption = processor.batch_decode(gen_only_ids, skip_special_tokens=True)[0]

    input_tokens = int(prompt_len)
    output_tokens = int(gen_only_ids.shape[1])
    total_tokens = input_tokens + output_tokens
    print(f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total: {total_tokens}")

    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = processor.tokenizer.eos_token_id
    gen_token_ids = gen_only_ids[0].tolist()
    if isinstance(eos_token_id, (list, tuple, set)):
        finished_with_stop = any(tok in eos_token_id for tok in gen_token_ids)
    elif eos_token_id is None:
        finished_with_stop = False
    else:
        finished_with_stop = eos_token_id in gen_token_ids
    finish_reason = "stop" if finished_with_stop else "length"
    if finish_reason == "length":
        print("  Warning: Caption truncated due to max_new_tokens limit!")

    captions_by_name[video_name] = caption
    write_captions_ordered()

    print(f"  Completed in {elapsed_time:.2f}s")
    print(f"  Caption length: {len(caption)} characters")
    print(f"  Finish reason: {finish_reason}")
    return True

for idx, video_path in enumerate(video_entries, 1):
    try:
        video_name = video_path.relative_to(base_dir).as_posix()
    except ValueError:
        video_name = video_path.as_posix()

    if not video_path.exists():
        print(f"\n[{idx}/{len(video_entries)}] Missing file, skipping: {video_path}")
        log_failed(video_name, "missing_file")
        continue

    if video_name in processed_videos:
        print(f"\n[{idx}/{len(video_entries)}] Skipping (already processed): {video_name}")
        continue

    print(f"\n[{idx}/{len(video_entries)}] Processing: {video_name}")

    try:
        success = run_one_video(video_path, video_name)
        if success:
            processed_videos.add(video_name)
    except Exception as e:
        print(f"  Error processing {video_name}: {e}")
        log_failed(video_name, str(e))
        continue

print(f"\nAll videos processed. Results saved to {output_file}")
