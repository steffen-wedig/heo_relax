from ase import Atoms
from ase.filters import ExpCellFilter
from ase.optimize import FIRE, LBFGSLineSearch
from mace.calculators import MACECalculator
from ase.io import Trajectory

def relax_material_FIRE(
    material: Atoms,
    mace_calc: MACECalculator,
    ftol: float = 0.05,
    max_N_optimization_steps: int = 100,
):
    """
    Performs FIRE relaxation with variable cell size - does not allow for shearing the cell though.
    """
    material.calc = mace_calc
    cell_filter = ExpCellFilter(material, hydrostatic_strain=True)
    opt = FIRE(cell_filter)
    opt.run(fmax=ftol, steps=max_N_optimization_steps)


def relax_material_LBFGS(
    material: Atoms,
    mace_calc: MACECalculator,
    ftol: float = 0.05,
    max_N_optimization_steps: int = 100,
    trajectory_file: str | None = None
):
    """
    Performs LBFGS relaxation with variable cell size - does not allow for shearing the cell though.
    """

    material.calc = mace_calc
    cell_filter = ExpCellFilter(material)
    opt = LBFGSLineSearch(cell_filter, trajectory = trajectory_file)
    opt.run(fmax=ftol, steps=max_N_optimization_steps)
    N_steps = opt.get_number_of_steps()
    converged_flag = opt.converged()

    return converged_flag, N_steps
