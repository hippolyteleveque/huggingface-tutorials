from datasets import load_dataset

data_files = {"train": "data/drugsComTrain_raw.tsv",
              "test": "data/drugsComTest_raw.tsv"}

drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)


def filter_nones(x):
    return x["condition"] is not None

# Normalize the condition


def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
print(drug_dataset["train"][:3])
