import math
import os
import random
from itertools import accumulate
from pathlib import Path

from ase import Atoms
from ase.io import write


def split_train_val_test_dataset(
    materials : list[Atoms], train_valid_test_ratio: list[float], shuffle=False
) -> tuple[list[Atoms], list[Atoms], list[Atoms]]:

    assert sum(train_valid_test_ratio) == 1
    assert len(train_valid_test_ratio) == 3

    split_lengths = [math.floor(ratio * len(materials)) for ratio in train_valid_test_ratio]

    split_indices = list(accumulate(split_lengths))

    if shuffle:
        random.shuffle(materials)

    training_data = materials[:split_indices[0]]
    validation_data = materials[split_indices[0]:split_indices[1]]
    test_data = materials[split_indices[1]:]

    return training_data, validation_data, test_data


def write_ase_atoms_to_file(filename: Path, ase_atoms: list[Atoms]):
    directory = filename.parent
    os.makedirs(directory, exist_ok=True)

    write(filename, ase_atoms)
