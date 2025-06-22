import evaluate
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm

# Load SAMSum dataset
dataset = load_dataset("samsum")
test_data = dataset["test"]

# Load T5 model and tokenizer
model_name = "./nash_out/t5-small/SAMSUM/NASH/SAMSUM_nash_unif_0.3_3/int/FT/best"

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

tokenizer = T5Tokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load ROUGE metric
rouge = evaluate.load("rouge")

def generate_summary(dialogue):
    input_text = "summarize: " + dialogue
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest", max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=64)
    summary = tokenizer.decode(summary_ids[0].flatten(), skip_special_tokens=True)
    return summary

references = []
predictions = []

for example in tqdm(test_data, desc="Evaluating"):
    dialogue = example["dialogue"]
    summary = example["summary"]
    pred_summary = generate_summary(dialogue)
    references.append(summary)
    predictions.append(pred_summary)
    
    if len(predictions) == 1:
        print("\nSample Dialogue:\n", dialogue)
        print("Reference Summary:\n", summary)
        print("Predicted Summary:\n", pred_summary)

# Compute ROUGE scores
results = rouge.compute(predictions=predictions, references=references)
for key, value in results.items():
    print(f"{key}: {value:.4f}")