import argparse, os, json, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

def load_jsonl(path): return load_dataset("json", data_files=path, split="train")

def fmt(example, tok):
    return tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

def main():
    os.makedirs("outputs/tinyllama-1.1b-qlora-tuned", exist_ok=True)
    os.makedirs("offload", exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization_config=bnb,
        device_map="auto",
        offload_folder="offload",
        dtype=torch.float16,
    )

    lcfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dataset = load_jsonl("data/qa_tracks_recommender.jsonl").map(lambda ex: {"text": fmt(ex, tok)})
    train_ds = dataset.select(range(9000))
    eval_ds  = dataset.select(range(9000, len(dataset)))

    targs = SFTConfig(
        output_dir="outputs/tinyllama-1.1b-qlora-style",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=.0001,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=.03,
        weight_decay=.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        dataset_text_field="text",
        packing=True,
        max_length=512
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        peft_config=lcfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=targs,
    )

    trainer.train()
    trainer.save_model("outputs/tinyllama-1.1b-qlora-style")
    tok.save_pretrained("outputs/tinyllama-1.1b-qlora-style")
    print("Adapter saved in: outputs/tinyllama-1.1b-qlora-style")

if __name__ == "__main__":
    main()
