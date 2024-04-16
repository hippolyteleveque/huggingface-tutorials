from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

camembert_fill_mask = pipeline("fill-mask", "camembert-base")
results = camembert_fill_mask("Le camembert est <mask>")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
