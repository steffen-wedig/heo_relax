
import numpy as np
import polars as pl
from ase import Atoms
from datasets import load_dataset

from heos_relax.data_handling.element_sets import (
    METALS_AND_OXYGEN,
)


class LeMaterialsDataset:
    @staticmethod
    def get_streaming_dataset():
        dataset = load_dataset(
            "LeMaterial/LeMat-Bulk", "compatible_pbe", split="train", streaming=True
        ).with_format("polars")

        return dataset

    @staticmethod
    def filter_elements(df_batch: pl.DataFrame, min_nelements: int) -> pl.DataFrame:

        #samples have to contain oxygen and only other metals
        return df_batch.filter(
            pl.col("nelements") >= min_nelements,
            pl.col("elements").list.contains("O"),
            pl.col("elements")
            .list.eval(pl.element().is_in(METALS_AND_OXYGEN))
            .list.all(),
            pl.col("forces").list.len() > 0, # we want to train on forces so enforce that they are present in the data
        )

    def filter_configurational_entropy():
        # Conventionally, HEOs should be differentiated by their configurational entropy
        raise NotImplementedError

    def load_heos_from_lemat(self, min_nelements: int = 5, batch_size: int = 1024):
        """
        Returns the raw heos from lemat dataset.
        """

        ds = self.get_streaming_dataset()

        df_heo_materials = pl.DataFrame()
        for df_batch in ds.iter(batch_size=batch_size):
            filtered_by_element = LeMaterialsDataset.filter_elements(
                df_batch, min_nelements
            )

            if not filtered_by_element.is_empty():
                df_heo_materials.vstack(filtered_by_element, in_place=True)

        self.heo_materials_df = df_heo_materials

        return df_heo_materials

    @staticmethod
    def check_data_source(df: pl.DataFrame):
        # counts from which original dataset the samples are sourced

        counts = df.select(
            [
                # .str.contains() returns a Boolean Series; sum() treats True as 1, False as 0
                pl.col("immutable_id").str.contains("mp").sum().alias("mp_count"),
                pl.col("immutable_id").str.contains("oqmd").sum().alias("oqmd_count"),
                pl.col("immutable_id").str.contains("agm").sum().alias("agm_count"),
            ]
        )
        return counts

    def convert_df_to_ase_atoms(self):
        ase_atoms = []

        for mat in self.heo_materials_df.iter_rows(named=True):
            # Creates ase Atoms
            atoms = Atoms(
                symbols=mat["species_at_sites"],
                positions=mat["cartesian_site_positions"],
                pbc=True,
                cell=mat["lattice_vectors"],
            )

            # Add the Reference values to the atoms object, and name them appropriately for mace
            atoms.info["DFT_energy"] = mat["energy"]
            atoms.arrays["DFT_forces"] = np.array(mat["forces"])
            ase_atoms.append(atoms)

        self.atoms = ase_atoms
        return ase_atoms
