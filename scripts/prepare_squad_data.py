import json, random, re, argparse
from datasets import load_dataset

random.seed(42)

SYS_EN = "You are a concise and technical teaching assistant. Answer in English."

def to_messages(instruction, response, system=SYS_EN):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction.strip()},
        {"role": "assistant", "content": response.strip()}
    ]

def clean(s): return re.sub(r"\s+", " ", s or "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out_train", default="../data/squad_1k_chat.jsonl")
    args = ap.parse_args()

    ds = load_dataset("squad")["train"]
    pool = []
    for ex in ds:
        instr, resp = clean(ex.get("question")), clean(ex.get("answers").get("text")[0])
        if instr and resp:
            pool.append((instr, resp))
    random.shuffle(pool)
    pool = pool[:args.samples]

    with open(args.out_train, "w", encoding="utf-8") as f:
        for instr, resp in pool:
            f.write(json.dumps({"messages": to_messages(instr, resp)}, ensure_ascii=False)+"\n")

    print("Wrote :", args.out_train)

if __name__ == "__main__":
    main()
