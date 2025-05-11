import ase
from pathlib import Path
import os
from ase import Atoms
from ase.io import write



def train_val_test_split(split_ratios):
    pass



def write_ase_atoms_to_file(filename : Path, ase_atoms : list[Atoms]):

    directory = filename.parent
    os.makedirs(directory, exist_ok=True)

    write(filename, ase_atoms)