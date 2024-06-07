import os
from datasets import load_dataset

def load_stochastic_books(file_path):
    data_files = {"train": [file_path]}
    dataset = load_dataset('text', data_files=data_files)
    return dataset
