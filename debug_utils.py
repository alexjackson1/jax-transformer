from typing import Dict, Union
from jaxtyping import Array

Data = Union[Array, Dict[str, "Data"]]


def print_nested_structure(data: Data, indent: int = 0):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"\n{'  ' * indent}{key}:", end=" ")
            print_nested_structure(value, indent + 1)
    # else if tuple
    elif isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            print(f"\n{'  ' * indent}{i}:", end=" ")
            print_nested_structure(value, indent + 1)
    elif "shape" in dir(data):
        print(f"{data.shape}", end="")
    else:
        print(data)
        raise TypeError("Input must be an array or a dictionary")

    if indent == 0:
        print("")
