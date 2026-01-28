import time
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

# Base directory containing video subdirectories
base_dir = Path("/home/dataset/video_eval")

levels = ["L1", "L2", "L3", "L4", "L5"]
durations = ["short", "medium", "long"]

# Output file for captions
output_file = "captions_qwen3-30B-A3B-Instruct.jsonl"

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

for level in levels:
    for duration in durations:
        video_dir = base_dir / level / duration

        if not video_dir.exists():
            print(f"Warning: Directory {video_dir} does not exist. Skipping...")
            continue

        print(f"\nProcessing {level}/{duration} videos...")

        video_files = sorted(video_dir.glob("*.mp4"))

        print(f"Found {len(video_files)} videos in {level}/{duration}")

        for idx, video_path in enumerate(video_files, 1):
            video_name = video_path.relative_to(base_dir).as_posix()

            # Skip if already processed
            if video_name in processed_videos:
                print(f"\n[{idx}/{len(video_files)}] Skipping (already processed): {video_name}")
                continue

            video_url = f"file://{video_path.absolute()}"

            print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")

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
                            "text": "Describe the video in detail.",
                        }
                    ]
                }
            ]

            try:
                start = time.time()
                response = client.chat.completions.create(
                    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,
                    top_p=1,
                    seed=42,
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
                    print(f"  Warning: Caption truncated due to max_tokens limit!")

                result = {
                    "video_name": video_name,
                    "caption": caption,
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                print(f"  Completed in {elapsed_time:.2f}s")
                print(f"  Caption length: {len(caption)} characters")
                print(f"  Finish reason: {finish_reason}")

            except Exception as e:
                error_msg = str(e)
                print(f"  Error processing {video_name}: {error_msg}")
                continue

print(f"\nAll videos processed. Results saved to {output_file}")
