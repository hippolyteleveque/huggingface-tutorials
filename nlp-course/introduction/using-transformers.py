from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# Loading the tokenizer for the defined checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# "PT" stands for pytorch tensors
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# Loading the model from the defined checkpoint
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)

# Loading the whole model including the head that has specialized in a specific task
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs_logits = model(**inputs)

# Computes the probabilities associated with each classes
predictions = torch.nn.functional.softmax(outputs_logits.logits, dim=-1)

if __name__ == "__main__":
    print(f"Tokenized inputs: {inputs}")
    print("\n\n")
    print(f"Hidden state shape: {outputs.last_hidden_state.shape}")
    print("\n\n")
    print(f"Outputs logits : {outputs_logits.logits}")
    print("\n\n")
    print(f"Predictions: {predictions}")
    print("\n\n")
    print(f"Classes: {model.config.id2label}")
