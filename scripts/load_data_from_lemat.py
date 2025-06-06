
from pathlib import Path

from heos_relax.data_handling.datasets import (
    split_train_val_test_dataset,
    write_ase_atoms_to_file,
)
from heos_relax.data_handling.lemat import LeMaterialsDataset
from heos_relax.data_handling.visualisation import DatasetVisualisation


import argparse

parser = argparse.ArgumentParser(description="Load the LeMaterials Training Data")
parser.add_argument(
    "--dir",
    dest="dir",
    type=str,
    required=True,
    help="Path to the data directory"
)
args = parser.parse_args()
output_dir = Path(args.dir)

dataset = LeMaterialsDataset()
dataset.load_heos_from_lemat(min_nelements=5, batch_size=1024)
atoms = dataset.convert_df_to_ase_atoms()

dv = DatasetVisualisation(heo_dataframe=dataset.heo_materials_df, heo_atoms=atoms)
dv.plot(output_dir= output_dir)

train_split, validation_split, test_split = split_train_val_test_dataset(
    atoms, train_valid_test_ratio=[0.7, 0.2, 0.1], shuffle=True
)


write_ase_atoms_to_file(output_dir / "train.xyz", train_split)
write_ase_atoms_to_file(output_dir / "valid.xyz", validation_split)
write_ase_atoms_to_file(output_dir / "test.xyz", test_split)
