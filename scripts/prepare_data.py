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
    ap.add_argument("--dolly_samples", type=int, default=2000)
    ap.add_argument("--out_train", default="../data/dolly_2k_chat.jsonl")
    ap.add_argument("--out_eval", default="../data/val_small.jsonl")
    args = ap.parse_args()

    ds = load_dataset("databricks/databricks-dolly-15k")["train"]
    pool = []
    keep = {"open_qa","brainstorming","question_answering","closed_qa","information_extraction","summarization"}
    for ex in ds:
        if ex.get("category") in keep:
            instr, resp = clean(ex.get("instruction")), clean(ex.get("response"))
            if instr and resp:
                pool.append((instr, resp))
    random.shuffle(pool)
    pool = pool[:args.dolly_samples]

    with open(args.out_train, "w", encoding="utf-8") as f:
        for instr, resp in pool:
            f.write(json.dumps({"messages": to_messages(instr, resp)}, ensure_ascii=False)+"\n")

    eval_prompts = [
        {"messages": to_messages("Explain self-attention very briefly (4 sentences max).","")},
        {"messages": to_messages("Gives a QLoRA checklist on 4GB GPU, 5 bullet-points.","")},
        {"messages": to_messages("Rewrite in a concise and technical way: « We change the tone without losing obedience to the instructions. »","")},
    ]
    with open(args.out_eval, "w", encoding="utf-8") as f:
        for ex in eval_prompts:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")

    print("Wrote :", args.out_train, args.out_eval)

if __name__ == "__main__":
    main()
