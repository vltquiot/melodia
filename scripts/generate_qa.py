from openai import OpenAI
import json, time, random

client = OpenAI(api_key="") #API Key

with open("configs/prompt_for_qa.txt") as f:
    base_prompt = f.read()

with open("data/tracks.jsonl") as f:
    tracks = [json.loads(line) for line in f]

sampled_tracks = random.sample(tracks, 1000)

out = open("data/qa_tracks_recommender.jsonl", "w")

for i, track in enumerate(sampled_tracks):
    messages = [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": json.dumps(track, ensure_ascii=False)}
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        response_content = completion.choices[0].message.content

        # Strip any potential markdown code blocks
        response_content = response_content.strip()
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.startswith("```"):
            response_content = response_content[3:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()	

        qa_pairs = json.loads(response_content)
        
        for qa in qa_pairs:
            out.write(json.dumps(qa, ensure_ascii=False) + "\n")
        
        print(f"{i+1}/{len(sampled_tracks)} done - {len(qa_pairs)} QA pairs generated")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing response for track {i+1}: {e}")
        print(f"Response was: {response_content}")
    except Exception as e:
        print(f"Error processing track {i+1}: {e}")

out.close()
print(f"\nTotal: {1000 * 10} QA pairs expected in data/qa_tracks_recommender.jsonl")
