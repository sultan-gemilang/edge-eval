import random
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import difflib
import argparse

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load model and tokenizer
parser = argparse.ArgumentParser(description="Evaluate T5 model on CommonsenseQA.")
parser.add_argument(
    "--model_name",
    type=str,
    default="./nash_out/t5-small/SAMSUM/NASH/SAMSUM_nash_unif_0.3_3/int/FT/best",
    help="Path or name of the pretrained T5 model"
)
args = parser.parse_args()
model_name = args.model_name
tokenizer = T5Tokenizer.from_pretrained(model_name)

if "nash" in model_name.lower():
    from models.modeling_t5 import NashT5ForConditionalGeneration
    from utils.utils import *
    from utils.nash_utils import *

    zs_path = '/'.join(model_name.split('/')[:-2])
    zs = load_zs(zs_path)
    model = load_model(zs_path, NashT5ForConditionalGeneration, zs)
    model_path = f"{model_name}/pytorch_model.bin"
    trained_weight = torch.load(model_path, map_location='cpu')
    model.load_state_dict(trained_weight)
else:
    model = T5ForConditionalGeneration.from_pretrained(model_name)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CommonsenseQA validation set
dataset = load_dataset("commonsense_qa", split="validation")

def format_example(example):
    # Format: "question: ... choices: (A) ... (B) ... (C) ... (D) ... (E) ..."
    choices = example["choices"]["text"]
    choices_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    return f"question: {example['question']} choices: {choices_str}"

def match_choice(pred, choices, labels=None):
    pred = pred.strip().lower()
    # Handle letter-only answers (e.g., "A", "B", etc.)
    if labels is not None:
        for idx, label in enumerate(labels):
            if pred == label.lower() or pred == f"({label.lower()})":
                return idx

    # Fuzzy string matching
    scores = [difflib.SequenceMatcher(None, pred, c.lower()).ratio() for c in choices]
    max_score = max(scores)
    if max_score > 0.6:  # threshold for reasonable match
        return scores.index(max_score)

    # Fallback: word overlap
    pred_words = set(pred.split())
    overlap_scores = [len(pred_words & set(c.lower().split())) for c in choices]
    return overlap_scores.index(max(overlap_scores))

correct = 0
total = 0

for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
    input_text = format_example(example)
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=8)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    pred_idx = match_choice(pred, choices, labels)
    gold_idx = labels.index(example["answerKey"])
    
    # Print a sample question, choices, prediction, and gold answer
    if i <= 5:
        print("-" * 50)
        print("\nSample Question:", example["question"])
        print("Choices:")
        for label, choice in zip(example["choices"]["label"], choices):
            print(f"  {label}: {choice}")
        print("Model's Answer:", choices[pred_idx])
        print("Gold Answer:", choices[gold_idx])
        print("Raw Model Output:", pred)
        print("-" * 50)
    
    if pred_idx == gold_idx:
        correct += 1
    total += 1

accuracy = (correct / total)*100
print(f"Accuracy: {accuracy:.4f}%")