import matplotlib.pyplot as plt
import numpy as np
import polars as ps
from ase import Atoms
from ase.visualize.plot import plot_atoms
from elementembeddings.composition import composition_featuriser
from umap import UMAP
from importlib import resources

output_dir = resources.files("output")


class DatasetVisualisation:
    def __init__(self, heo_dataframe: ps.DataFrame, heo_atoms: list[Atoms]):
        self.atoms = heo_atoms
        self.df = heo_dataframe

    def _plot_elemental_distribution(self):
        elements_exploded = self.df.explode("elements")

        counts = (
            elements_exploded.group_by("elements")
            .count()
            .sort(by="count", descending=True)
        )

        fig = plt.figure()
        fig.set_figwidth(14)
        plt.bar(counts["elements"].to_list(), counts["count"].to_list())
        plt.xlabel("Element")
        plt.ylabel("Frequency of occuring in the dataset")
        plt.tight_layout()

        return fig

    def _plot_umap_projection(self):
        embeddings = composition_featuriser(
            data=self.df["chemical_formula_reduced"].to_list(),
            embedding="magpie",
        )

        print(len(embeddings))
        umap_projection = UMAP(n_components=2)
        reduced_dims = umap_projection.fit_transform(embeddings)

        fig = plt.figure()
        plt.scatter(reduced_dims[:, 0], reduced_dims[:, 1])
        plt.xlabel("Reduced Dim 0")
        plt.ylabel("Reduced Dim 1")
        plt.tight_layout()

        return fig

    def _plot_sample_structures(self, N_samples: int = 10):
        drawn_materials_idx = np.random.choice(np.arange(len(self.atoms)),size = N_samples,replace = False)


        fig, axes = plt.subplots(N_samples, 1)
        fig.set_figheight(5*N_samples)
        for plot_idx, mat_idx in enumerate(drawn_materials_idx):
            axes[plot_idx].set_title(self.atoms[mat_idx].get_chemical_formula())
            plot_atoms(self.atoms[mat_idx], axes[plot_idx], rotation=('90x,45y,45z'))

        return fig


    def _plot_configurational_entropy_distribution(self):
        pass

    def _plot_n_elements_histogram(self):
        counts = self.df.group_by("nelements").count().sort(by="nelements")

        fig = plt.figure()
        plt.bar(counts["nelements"].to_list(), counts["count"].to_list())
        plt.xlabel("Number of constituent elements")
        plt.ylabel("Frequency of occuring in the dataset")
        plt.yscale("log")
        plt.xticks(
            ticks=counts["nelements"].cast(ps.Int32).to_list(),
            labels=counts["nelements"].cast(ps.Int32).to_list(),
        )
        plt.tight_layout()
        return fig

    def _plot_unit_cell_size_distribution():
        raise NotImplementedError

    def _plot_space_group_distiribution():
        raise NotImplementedError

    def plot(self):
        figs: dict[str : plt.Figure] = {}

        figs["umap_projection"] = self._plot_umap_projection()
        figs["nelements_histogram"] = self._plot_n_elements_histogram()
        figs["elemental_distribution"] = self._plot_elemental_distribution()
        figs["sample_structures"] = self._plot_sample_structures()

        for fig_name, fig in figs.items():
            fig.savefig(output_dir / f"{fig_name}.pdf")
