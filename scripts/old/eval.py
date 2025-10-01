import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROMPTS = [
    "Explain self-attention very briefly (4 sentences max).",
    "Gives a QLoRA checklist on 4GB GPU, 5 bullet-points.",
    "Rewrite in a concise and technical way: « We change the tone without losing obedience to the instructions. »"
]

SYS = "You are a concise and technical teaching assistant. Answer in English."

def chat(tok, msgs, add_gen=True):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)

def run(model, tok, prompt, max_new=192):
    msgs = [{"role":"system","content":SYS}, {"role":"user","content":prompt}]
    text = chat(tok, msgs, True)
    inps = tok([text], return_tensors="pt").to(model.device)
    out = model.generate(**inps, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)

def load_base(mid):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    return m, tok

def load_adapter(mid, adir):
    m, t = load_base(mid)
    m = PeftModel.from_pretrained(m, adir)
    return m, t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", default="outputs/tinyllama-1.1b-qlora-style")
    ap.add_argument("--out", default="eval/report.jsonl")
    args = ap.parse_args()

    base_m, base_t = load_base(args.base_model)
    ada_m,  ada_t  = load_adapter(args.base_model, args.adapter)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in PROMPTS:
            r0 = run(base_m, base_t, p)
            r1 = run(ada_m,  ada_t, p)
            f.write(json.dumps({"prompt": p, "base": r0, "adapter": r1}, ensure_ascii=False)+"\n")
    print("Report:", args.out)
