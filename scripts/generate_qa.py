import json
import random
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams

TRACKS_FILE = Path("data/tracks.jsonl")
PROMPT_FILE = Path("configs/prompt_for_qa.txt")
OUTPUT_FILE = Path("data/qa_tracks_recommender.jsonl")

MODEL_NAME = "mistral-7b-instruct-v0.2"
NUM_SAMPLES = 1000 #1000*10 pairs => 10k QA 
MAX_TOKENS = 1024

with TRACKS_FILE.open("r", encoding="utf-8") as f:
    tracks = [json.loads(line) for line in f]

sampled_tracks = random.sample(tracks, min(NUM_SAMPLES, len(tracks)))

with PROMPT_FILE.open("r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

print("Loading model locally...")
llm = LLM(model=MODEL_NAME)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=MAX_TOKENS
)

with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
    for track in tqdm(sampled_tracks, desc="Generating QA"):
        # Add metadata to the end of the base prompt
        prompt = f"{base_prompt}\n\n{json.dumps(track, ensure_ascii=False)}"

        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text.strip()

        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    out_f.write(line + "\n")
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

print(f"QA dataset written to {OUTPUT_FILE}")
