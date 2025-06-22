import evaluate
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import time
import numpy as np
import argparse

from models.modeling_t5 import NashT5ForConditionalGeneration
from utils.utils import *
from utils.nash_utils import *

def tokenize_input(tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest", max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    return input_ids, attention_mask

def main():
    parser = argparse.ArgumentParser(description="T5 SAMSum Evaluation with Latency Metrics")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Model name or path")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum number of tokens to generate for summary")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to evaluate (default: all)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    args = parser.parse_args()

    # Load SAMSum dataset
    dataset = load_dataset("knkarthick/samsum")
    test_data = dataset["test"]
    if args.max_samples is not None:
        test_data = test_data.select(range(args.max_samples))

    # Load T5 model and tokenizer
    model_name = args.model_name

    if "nash" in model_name.lower():
        zs_path = '/'.join(model_name.split('/')[:-2])
        zs = load_zs(zs_path)
        model = load_model(zs_path, NashT5ForConditionalGeneration, zs)
        model_path = f"{model_name}/pytorch_model.bin"
        trained_weight = torch.load(model_path, map_location='cpu')
        model.load_state_dict(trained_weight)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)

    # Load ROUGE metric
    rouge = evaluate.load("rouge")

    references = []
    predictions = []

    # Latency metrics
    ttft_list = []
    tgt_list = []
    tpot_list = []
    tps_list = []
    gen_list = []

    max_length = args.max_length
    num_samples = len(test_data)

    # -------- Warmup --------
    print(f"------ Warmup ({args.warmup} runs) ------")
    warmup_dialogue = test_data[0]["dialogue"]
    warmup_input_text = "summarize: " + warmup_dialogue
    for _ in tqdm(range(args.warmup), desc="Warmup"):
        input_ids, attention_mask = tokenize_input(tokenizer, warmup_input_text, device)
        with torch.no_grad():
            _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)
            _ = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)

    # -------- TTFT Evaluation --------
    for idx, example in enumerate(tqdm(test_data, desc="Evaluating TTFT")):
        dialogue = example["dialogue"]
        summary = example["summary"]

        # Prepare input
        input_text = "summarize: " + dialogue
        input_ids, attention_mask = tokenize_input(tokenizer, input_text, device)

        # TTFT: Time to First Token
        with torch.no_grad():
            start_ttft = time.perf_counter()
            _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1)
            end_ttft = time.perf_counter()
        ttft = end_ttft - start_ttft
        ttft_list.append(ttft)
        
        if idx == 0:
            print("\nSample Dialogue:\n", dialogue)
            print("Reference Summary:\n", summary)
            print(f"TTFT Time: {ttft:.4f} seconds")
    
    # -------- TGT & TPOT Evaluation --------
    for idx, example in enumerate(tqdm(test_data, desc="Evaluating TGT & TPOT")):
        # TGT & TPOT: Full summary generation
        
        input_text = "summarize: " + example["dialogue"]
        summary = example["summary"]
        
        start_tgt = time.perf_counter()
        input_ids, attention_mask = tokenize_input(tokenizer, input_text, device)
        
        with torch.no_grad():
            start_tpot = time.perf_counter()
            summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
            end_tpot = time.perf_counter()
            generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            end_tgt = time.perf_counter()

        gen_token_len = summary_ids.shape[1]
        tpot_time = end_tpot - start_tpot
        tgt_time = end_tgt - start_tgt
        tps = gen_token_len / tpot_time if tpot_time > 0 else float('inf')

        tgt_list.append(tgt_time)
        tpot_list.append(tpot_time)
        tps_list.append(tps)
        gen_list.append(gen_token_len)

        references.append(summary)
        predictions.append(generated_text)

        if idx == 0:
            print("\nSample Dialogue:\n", example["dialogue"])
            print("Reference Summary:\n", summary)
            print("Predicted Summary:\n", generated_text)
            print(f"TGT Time: {tgt_time:.4f} seconds, TPOT Time: {tpot_time:.4f} seconds, TPS: {tps:.2f} tokens/sec")
            print(f"Generated Token Length: {gen_token_len}")

    # Compute ROUGE scores
    results = rouge.compute(predictions=predictions, references=references)
    
    print("\n------ ROUGE Results ------")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # Print latency metrics
    print("\n------ Latency Results ------")
    print(f"Samples evaluated: {num_samples}")
    print(f"Warmup runs: {args.warmup}")
    print(f"avg TTFT \t: {np.mean(ttft_list) * 1000:.2f} ms")
    print(f"avg TGT \t: {np.mean(tgt_list) * 1000:.2f} ms")
    print(f"avg TPOT \t: {np.mean(tpot_list) * 1000:.2f} ms")
    print(f"avg TPS \t: {np.mean(tps_list):.2f} tokens/sec")
    print(f"avg gen_len \t: {np.mean(gen_list):.2f} tokens")

if __name__ == "__main__":
    main()