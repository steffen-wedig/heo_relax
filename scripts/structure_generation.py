import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

from heos_relax.structure_generation import RandomHighEntropyOxideStructureGeneration

generator = RandomHighEntropyOxideStructureGeneration(
    space_group=225, composition="Mg1Ni1Cu1Co1Zn1O5"
).generate_structures(10)

structures = generator

for i, atoms in enumerate(structures):
    fig = plt.figure()
    plot_atoms(atoms, plt.gca(), rotation=("90x,45y,45z"))
    fig.savefig(f"random_generate_heostruct_{i}.png")
