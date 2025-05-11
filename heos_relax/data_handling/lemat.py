import logging

import matplotlib.pyplot as plt
import polars as ps
from ase import Atoms
from datasets import load_dataset
import numpy as np
from heos_relax.data.element_sets import METALS, METALS_AND_OXYGEN, OTHER_ANIONS

logger = logging.getLogger(__name__)


def filter_elements(df_batch: ps.DataFrame, min_nelements: int) -> ps.DataFrame:
    return df_batch.filter(
        ps.col("nelements") >= min_nelements,
        ps.col("elements").list.contains("O"),
        ps.col("elements").list.eval(ps.element().is_in(METALS_AND_OXYGEN)).list.all(),
        ps.col("forces").list.len() > 0,
    )


def filter_configurational_entropy():
    pass


def load_heos_from_lemat(min_nelements: int = 5, batch_size: int = 1024):
    """
    Returns the raw heos from lemat dataset.
    """

    ds = load_dataset(
        "LeMaterial/LeMat-Bulk", "compatible_pbe", split="train", streaming=True
    ).with_format("polars")

    df_heo_materials = ps.DataFrame()
    for df_batch in ds.iter(batch_size=batch_size):
        filtered_by_element = filter_elements(df_batch, min_nelements)

        if not filtered_by_element.is_empty():
            df_heo_materials.vstack(filtered_by_element, in_place=True)

    logger.info(f"Loaded {len(df_heo_materials)} from LeMaterials")
    return df_heo_materials


def convert_df_to_ase_atoms(df_materials: ps.DataFrame):
    ase_atoms = []

    for mat in df_materials.iter_rows(named=True):
        # Creates ase Atoms
        atoms = Atoms(
            symbols=mat["species_at_sites"],
            positions=mat["cartesian_site_positions"],
            pbc=True,
            cell=mat["lattice_vectors"],
        )

        atoms.info["DFT_energy"] = mat["energy"]
        atoms.arrays["DFT_forces"] = np.array(mat["forces"])
        ase_atoms.append(atoms)

    return ase_atoms
