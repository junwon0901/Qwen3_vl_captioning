import time
import json
import random
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

base_dir = Path("/home/dataset/video_eval")
video_index_file = base_dir / "VE-500.json"

# Output file for captions
output_file = "captions_qwen3-8B-instruct.jsonl"
max_retries_on_length = 10

# Load already processed videos
processed_videos = set()
if Path(output_file).exists():
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    processed_videos.add(data.get("video_name"))
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

for idx, video_path in enumerate(video_entries, 1):
    if not video_path.exists():
        print(f"\n[{idx}/{len(video_entries)}] Missing file, skipping: {video_path}")
        continue

    try:
        video_name = video_path.relative_to(base_dir).as_posix()
    except ValueError:
        video_name = video_path.as_posix()

    # Skip if already processed
    if video_name in processed_videos:
        print(f"\n[{idx}/{len(video_entries)}] Skipping (already processed): {video_name}")
        continue

    video_url = f"file://{video_path.absolute()}"

    print(f"\n[{idx}/{len(video_entries)}] Processing: {video_name}")

            prompt_options = [
                "Describe this video in detail, focusing on the spatio-temporal dynamics. Describe exactly how objects and agents move, change, and occupy space over time within the scene.",
                "Give a detailed account of everything shown in the video, capturing all visible specifics. Describe events in the exact order they appear over time. Ensure that you describe the sequence of events exactly as they occur, without skipping any steps.",
                "Describe the video in detail, paying special attention to how objects and people interact with each other. Capture the precise timing and nature of every contact, movement, and reaction shown in the footage.",
                "Thoroughly describe the video’s visual narrative, capturing every visible detail from start to end. Emphasize how actions unfold and how the scene transitions logically over time.",
                "Provide a detailed, continuous description of everything visible in the video. Describe only what is directly visible. Do not guess, assume, or add anything that is not clearly shown (e.g., implied audio, invisible causes, or future events)"
            ]
            selected_prompt = random.choice(prompt_options)

            messages = [
                {
                    "role": "user",
                    "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": video_url
                    },
                        },
                        {
                            "type": "text",
                            "text": selected_prompt,
                        }
                    ]
                }
            ]

    try:
        attempt = 0
        while True:
            attempt += 1
            start = time.time()
            response = client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=messages,
                max_tokens=512,
            )
            elapsed_time = time.time() - start

            caption = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            usage = getattr(response, 'usage', None)
            if usage:
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                print(f"  Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total: {total_tokens}")

            if finish_reason == "length":
                print(f"  Warning: Caption truncated due to max_tokens limit! (retry {attempt}/{max_retries_on_length})")
                if attempt < max_retries_on_length:
                    continue
                print(f"  Giving up after {max_retries_on_length} retries due to repeated length truncation.")

            result = {
                "video_name": video_name,
                "caption": caption,
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"  Completed in {elapsed_time:.2f}s")
            print(f"  Caption length: {len(caption)} characters")
            print(f"  Finish reason: {finish_reason}")
            break

    except Exception as e:
        error_msg = str(e)
        print(f"  Error processing {video_name}: {error_msg}")
        continue

print(f"\nAll videos processed. Results saved to {output_file}")
