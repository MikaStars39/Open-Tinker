from .openr1 import load_openr1_dataset
from .numina import load_numina_dataset

def load_dataset(dataset_name_or_path: str, example_numbers: int = None):
    if "NuminaMath-TIR" in dataset_name_or_path:
        return load_numina_dataset(dataset_name_or_path, example_numbers)
    elif "OpenR1" in dataset_name_or_path:
        return load_openr1_dataset(dataset_name_or_path, example_numbers)