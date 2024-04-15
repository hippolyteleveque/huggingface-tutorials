from transformers import AutoTokenizer, AutoModelForSequenceClassification

# We get the Bert model trained by HF
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# We load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

print(model(**model_inputs).logits)
