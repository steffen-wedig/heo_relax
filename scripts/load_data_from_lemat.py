from heos_relax.data_handling.lemat import load_heos_from_lemat, convert_df_to_ase_atoms
from heos_relax.data_handling.datasets import write_ase_atoms_to_file
from pathlib import Path

heo_dataset = load_heos_from_lemat(5, 1024)
atoms = convert_df_to_ase_atoms(heo_dataset)


train_filename = Path("/home/steffen/projects/heos_relax/data/train.xyz")
write_ase_atoms_to_file(train_filename, atoms)
