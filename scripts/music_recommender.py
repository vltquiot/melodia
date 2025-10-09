import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "outputs/tinyllama-1.1b-qlora-style/"

TOP_K_RETRIEVAL = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Loading base model on {DEVICE}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Model loaded successfully!")
    return model, tokenizer

def format_prompt(query, tokenizer):

    messages = [
        {
            "role": "system",
            "content": "You are a music recommendation assistant. Use the provided context to answer questions about songs, artists, and music recommendations. Be helpful and informative."
        },
        {
            "role": "user",
            "content": query
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

def ask_question(query, model, tokenizer, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print(f"{'='*60}")
    
    if verbose:
        print("\nFormatting prompt with context...")
    
    prompt = format_prompt(query, tokenizer)
    
    if verbose:
        print("\nGenerating response...")
    
    response = generate_response(model, tokenizer, prompt)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Response:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}\n")
    
    return response

def main():
    print("\n" + "="*60)
    print("MUSIC RECOMMENDATION")
    print("="*60 + "\n")
    
    try:
        print("\nSTEP 1: Loading fine-tuned model...")
        model, tokenizer = load_model_and_tokenizer()
        
        print("\n" + "="*60)
        print("SETUP COMPLETE! Ready to answer questions.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_query = input("You: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not user_query:
                    continue
                
                ask_question(user_query, model, tokenizer)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nSomething went wrong. Please check your configuration.")


if __name__ == "__main__":
    main()
