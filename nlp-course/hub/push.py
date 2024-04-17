from transformers import TrainingArguments, AutoTokenizer, AutoModelForMaskedLM

training_args = TrainingArguments(
    "bert-finetuned-mrps", save_strategy="epoch", push_to_hub=True
)

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.push_to_hub("my-first-model")
tokenizer.push_to_hub("my-first-model")
