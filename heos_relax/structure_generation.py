import re

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from pyxtal import pyxtal

from heos_relax.data_handling.element_sets import METALS


class RandomHighEntropyOxideStructureGeneration:
    """
    Responsible for generating high entropy oxide configurations by randomly substituting ions from a parent crystal structure.
    Used as start of the relaxation.
    """

    def __init__(self, space_group, composition):
        self.space_group = space_group
        self.composition = composition

        self.random_number_generator = np.random.default_rng(0)

    def get_cation_elements_and_probabilties(self):
        pattern = r"([A-Z][a-z]?)(\d+(\.\d+)?)"
        matches = re.findall(pattern, self.composition)
        # keep only non-oxygen entries, convert counts to floats
        metals = {el: float(n) for el, n, _ in matches if el != "O"}
        assert all([element in METALS for element in metals.keys()])

        total = sum(metals.values())
        assert total != 0

        metals = {k: v / total for k, v in metals.items()}

        return list(metals.keys()), list(metals.values())

    def generate_structures(self, N_samples) -> list[Atoms]:
        xtal = pyxtal()
        xtal.from_random(group=225, species=["Ca", "O"], numIons=[4, 4])
        atoms = xtal.to_ase()
        atoms = make_supercell(atoms, P=np.eye(3) * 4)

        elements, probabilities = self.get_cation_elements_and_probabilties()
        structures = []

        for _ in range(N_samples):
            new_structure = atoms.copy()

            atomic_symbols = np.array(atoms.get_chemical_symbols())
            ca_mask = np.where(atomic_symbols == "Ca", True, False)
            N_placeholders = np.sum(ca_mask)

            new_species = self.random_number_generator.choice(
                a=elements, size=N_placeholders, p=probabilities
            )

            np.place(atomic_symbols, ca_mask, new_species)
            new_structure.set_chemical_symbols(atomic_symbols)
            structures.append(new_structure)

        return structures
