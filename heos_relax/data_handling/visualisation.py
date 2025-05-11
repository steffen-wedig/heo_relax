import polars as ps
from ase import Atoms
import matplotlib.pyplot as plt


class DatasetVisualisation:
    def __init__(self, heo_dataframe, heo_atoms):
        self.atoms = heo_atoms
        self.df = heo_dataframe

    def _plot_elemental_distribution(self):
        pass

    def _plot_umap_projection(self):
        pass

    def _plot_sample_structures(self):
        pass

    def _plot_configurational_entropy_distribution(self):
        pass