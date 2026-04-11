import os

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def get_dataset_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]