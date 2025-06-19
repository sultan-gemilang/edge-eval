import time
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import evaluate  # <-- new import

def main():
    # Load SAMSum dataset
    dataset = load_dataset("samsum")
    test_data = dataset["test"]

    # Load T5 model and tokenizer
    model_name = "./out-test/SAMSUM/SAMSUM_t5-base/best"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    # Metric for evaluation
    rouge = evaluate.load("rouge")  # <-- updated

    # Benchmarking
    start_time = time.time()
    predictions = []
    references = []

    for sample in test_data:
        input_text = "summarize: " + sample["dialogue"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        with torch.no_grad():
            summary_ids = model.generate(input_ids, max_length=60, num_beams=4, early_stopping=True)
        pred = tokenizer.decode(summary_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
        predictions.append(pred)
        # Ensure reference is a string, not a list
        reference = sample["summary"]
        if isinstance(reference, list):
            reference = reference[0]
        references.append(reference)

    # Compute ROUGE scores
    result = rouge.compute(predictions=predictions, references=references)
    elapsed = time.time() - start_time

    print("T5 on SAMSum Benchmark Results:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
    print(f"Total inference time: {elapsed:.2f} seconds")
    print(f"Average time per sample: {elapsed/len(test_data):.4f} seconds")

if __name__ == "__main__":
    main()