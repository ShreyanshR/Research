import lzma
import pickle
from typing import Any

def load_pickle(file_path: str) -> Any:
    """
    Load a pickled object from an lzma-compressed file.
    Args:
        file_path (str): Path to the .xz (lzma) pickle file.
    Returns:
        Any: The loaded Python object.
    """
    with lzma.open(file_path, "rb") as fp:
        return pickle.load(fp)

def save_pickle(file_path: str, obj: Any) -> None:
    """
    Save a Python object to an lzma-compressed pickle file.
    Args:
        file_path (str): Path to save the .xz (lzma) pickle file.
        obj (Any): The Python object to pickle.
    """
    with lzma.open(file_path, "wb") as fp:
        pickle.dump(obj, fp)

#class Alpha:l