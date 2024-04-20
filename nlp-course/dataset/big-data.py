from transformers import AutoTokenizer
from datasets import load_dataset


data_files = "https://huggingface.co/datasets/qualis2006/PUBMED_title_abstracts_2020_baseline/resolve/main/PUBMED_title_abstracts_2020_baseline.jsonl.zst"
# pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)

next(iter(pubmed_dataset_streamed))

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
# tokenized_dataset is also an iterable
print(type(tokenized_dataset))
# print(next(iter(tokenized_dataset)))

tokenized_dataset_batched = pubmed_dataset_streamed.map(
    lambda x: tokenizer(x["text"]), batched=True
)
