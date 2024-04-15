import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

# Basic training 
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]

tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples_base = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples_base.items() if k not in ["idx", "sentence1", "sentence2"]}

batch = data_collator(samples)


if __name__ == "__main__":
    print(f"The inputs : {inputs}")
    print("\n")
    print(f"Tokenized dataset: {tokenized_datasets}")
    print("\n")
    print(f"Sample base: {samples_base[:2]}")
