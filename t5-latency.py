from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime
from tqdm import tqdm

from models.modeling_t5 import NashT5ForConditionalGeneration
from utils.utils import *
from utils.nash_utils import *

import time
import torch
import random
import numpy as np
import argparse



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_model_and_tokenizer(model_name, device):
    if "nash" in model_name.lower():
        zs_path = '/'.join(model_name.split('/')[:-2])
        zs = load_zs(zs_path)
        model = load_model(zs_path, NashT5ForConditionalGeneration, zs)
        model_path = f"{model_name}/pytorch_model.bin"
        trained_weight = torch.load(model_path, map_location='cpu')
        model.load_state_dict(trained_weight)
    elif "t5-small" in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    elif "t5-base" in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    elif "t5-large" in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
    else:
        raise ValueError("Unsupported model type. Please use a T5 model or a NashT5 model.")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return model, tokenizer

def generate_text(model, input_ids, max_new_tokens, do_sample=False):
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )
    return outputs

def tokenize_input(tokenizer, input_text, device):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    return input_ids

def main():
    parser = argparse.ArgumentParser(description="T5 Latency Evaluation Script")
    parser.add_argument("--model_name", type=str, default="t5-large", help="Model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=1000, help="Number of measurement runs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu)")
    args = parser.parse_args()
    
    set_seed(42)
    device = args.device
    model_name = args.model_name
    max_new_tokens = args.max_new_tokens
    warmup = args.warmup
    runs = args.runs
    
    input_text = "translate English to German: Hello World! This is a latency test for T5 models. I hope it works well."

    print(f"model  used : {model_name}")
    print(f"device used : {device}")

    print("------ Loading model and tokenizer ------")
    model_start_time = datetime.now()
    
    model_start = time.perf_counter()
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    model_end = time.perf_counter()

    model_end_time = datetime.now()

    # Calculate prompt token length
    input_ids = tokenize_input(tokenizer, input_text, device)
    prompt_token_length = input_ids.shape[1]
    
    print(f"Prompt token length: {prompt_token_length}")
    
    
    tgt_list = []
    tpot_list = []
    tps_list = []
    ttft_list = []
    gen_list = []
    
    print("------ Warmup ------")
    for _ in tqdm(range(warmup)):
        input_ids = tokenize_input(tokenizer, input_text, device)
        _ = generate_text(model, input_ids, max_new_tokens, do_sample=False)

    print("------ Measuring TTFT latency ------")
    ttft_time = datetime.now()

    for i in tqdm(range(runs)):
        start_time = time.perf_counter()
        input_ids = tokenize_input(tokenizer, input_text, device)
        outputs = generate_text(model, input_ids, max_new_tokens=1, do_sample=False)

        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.perf_counter()

        gen_time = end_time - start_time
        ttft_list.append(gen_time)
        
        if i == 0:
            print(f"\nSample Input: {input_text}")
            print(f"Generated Text: {generated_text}")
        # print(f"Run {i+1}/{runs}: Generated text: {generated_text}, Time taken: {gen_time:.4f} seconds}")

    print("------ Measuring TGT & TPOT latency ------")
    tgt_pot_time = datetime.now()
    
    for i in tqdm(range(runs)):
        start_tgt = time.perf_counter()
        input_ids = tokenize_input(tokenizer, input_text, device)
        
        start_tpot = time.perf_counter()
        outputs = generate_text(model, input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        end_tpot = time.perf_counter()
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_tgtg = time.perf_counter()
        
        gen_token_len = outputs.shape[1]  # <-- Fix here
        
        tpot_time = end_tpot - start_tpot
        tgt_time = end_tgtg - start_tgt
        tps = gen_token_len / tpot_time if tpot_time > 0 else float('inf')
        
        
        tgt_list.append(tgt_time)
        tpot_list.append(tpot_time)
        tps_list.append(tps)
        gen_list.append(gen_token_len)
        
        if i == 0:
            print(f"\nSample Input: {input_text}")
            print(f"Generated Text: {generated_text}")
            print(f"TGT Time: {tgt_time:.4f} seconds, TPOT Time: {tpot_time:.4f} seconds, TPS: {tps:.2f} tokens/sec")
            print(f"Generated Token Length: {gen_token_len}")
    
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tgt = sum(tgt_list) / len(tgt_list)
    avg_tpot = sum(tpot_list) / len(tpot_list)
    avg_tps = sum(tps_list) / len(tps_list)
    avg_gen = sum(gen_list) / len(gen_list)
    
    print("\n------ Results ------")
    print(f"Model loading time: {model_end - model_start:.2f} seconds")
    
    print(f"Model loading started at: {model_start_time}")
    print(f"Model loading ended at: {model_end_time}")
    print(f"TTFT measurement started at: {ttft_time}")
    print(f"TGT & TPOT measurement started at: {tgt_pot_time}")

    print(f"\nRuns: {runs}, Warmup: {warmup}")
    print(f"avg TTFT \t: {avg_ttft * 1000:.2f} ms")
    print(f"avg TGT \t: {avg_tgt * 1000:.2f} ms")
    print(f"avg TPOT \t: {avg_tpot * 1000:.2f} ms")
    print(f"avg TPS \t: {avg_tps:.2f} tokens/sec")
    print(f"avg gen_len \t: {avg_gen:.2f} tokens")

    print("------ Summary ------")
    print(f"Total runs: {runs}, Warmup runs: {warmup}")
    print(f"Model: {model_name}, Device: {device}")
    print(f"Input text: {input_text}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Prompt token length: {prompt_token_length}")
    print("------ Finished ------")
if __name__ == "__main__":
    main()