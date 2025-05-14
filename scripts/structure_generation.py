import matplotlib.pyplot as plt

from heos_relax.data_handling.visualisation import plot_atom_with_species_legend
from heos_relax.structure_generation import RandomHighEntropyOxideStructureGeneration

generator = RandomHighEntropyOxideStructureGeneration( composition="Mg1Ni1Cu1Co1Zn1O5"
).generate_structures(1)

structures = generator

for i, atoms in enumerate(structures):
    fig, ax = plt.subplots()
    plot_atom_with_species_legend(atoms, ax)
    fig.tight_layout()
    fig.savefig(f"random_generated_heo_struct_{i}.png")
