from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

# Creating a non trained model from the default config
config = BertConfig()
model = BertModel(config)

# Loading model from a specific checkpoint
checkpoint = "bert-base-cased"
model = BertModel.from_pretrained(checkpoint)

# Save model
# model.save_pretrained("./")

tokenizer = BertTokenizer.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Whole flow packaged together
text = "Using transformers with huggingface is simple and quick."
packaged_ids = tokenizer(text)

# First, split the text into individual toekns
tokens = tokenizer.tokenize(text)

# Map the individual tokens to the vocabulary map
ids = tokenizer.convert_tokens_to_ids(tokens)

# Example of decoding
decoded_string = tokenizer.decode(ids)


if __name__ == "__main__": 
    print(f"The tokens generated by our tokenizer are the following: {tokens}")
    print("\n")
    print(f"The ids our tokenizer mapped the individual tokens to is the following: {ids}")
    print("\n")
    print(f"Here is how our text can be decoded to : {decoded_string}")