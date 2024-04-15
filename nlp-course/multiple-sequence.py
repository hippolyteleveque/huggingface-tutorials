import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)

ids = tokenizer.convert_tokens_to_ids(tokens)

# Don't forget that models expect a square input
input_ids = torch.tensor([ids])

# batching
batched_ids = [ids, ids]

inputs_double_ids = torch.tensor(batched_ids)

double_logits = model(inputs_double_ids).logits

# Handling inputs of different length
batched_ids = [[200, 200, 200], [200, 200]]

padding_id = 100

batched_ids = [[200, 200, 200], [200, 200, padding_id]]

sequence1_ids = torch.tensor([batched_ids[0]])

# Without the padding token
sequence2_ids = torch.tensor([batched_ids[1][:-1]])

seq1_res = model(sequence1_ids).logits
seq2_res = model(sequence2_ids).logits
seqs_res = model(torch.tensor(batched_ids)).logits

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))

if __name__ == "__main__":
    print(f"Result for the first sequence: {seq1_res}")
    print("\n")
    print(f"Result for the second sequence: {seq2_res}")
    print("\n")
    print(f"Result for the sequences: {seqs_res}")
    print("\n")
    print(f"Outputs with the attention mask: {outputs}")
    print("\n")
    print(f"Tokenized sequence: {tokenizer(sequence)}")
