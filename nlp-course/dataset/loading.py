from os import path
import sys
from datasets import load_dataset

FILE_PATH = path.dirname(sys.argv[0])
DATA_DIR = path.join(FILE_PATH, "..", "..", "data")

squad_it_dataset = load_dataset("json", data_dir=DATA_DIR, field="data")

print(squad_it_dataset)

