import argparse, os, json, yaml, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

def load_jsonl(path): return load_dataset("json", data_files=path, split="train")

def fmt(example, tok):
    return tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ft_qlora.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg.get("offload_folder","offload"), exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16
    )

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map=cfg.get("device_map","auto"),
        offload_folder=cfg.get("offload_folder","offload"),
        torch_dtype=torch.float16,
    )

    lcfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias=cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    train_ds = load_jsonl(cfg["dataset_train_path"]).map(lambda ex: {"text": fmt(ex, tok)})
    eval_ds  = load_jsonl(cfg["dataset_eval_path"]).map(lambda ex: {"text": fmt(ex, tok)})

    targs = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_total_limit=cfg["save_total_limit"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        lr_scheduler_type=cfg["lr_schedule"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        report_to=cfg.get("report_to", "none"),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        dataset_text_field="text",
        packing=cfg["packing"],
        max_length=cfg["max_seq_length"]
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
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print("Adapter saved in:", cfg["output_dir"])

if __name__ == "__main__":
    main()
