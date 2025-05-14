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


import numpy as np


def pretty_print_tasks(task_dicts, task_names=None):
    """
    Pretty-print a list of task result dictionaries to the terminal.

    Parameters:
    - task_dicts: list of dicts, each mapping task names to their metric dicts.
    - task_names: optional list of names for each set of tasks; if None, tasks will be numbered.
    """
    # Flatten input: allow list of dicts or single dict
    if isinstance(task_dicts, dict):
        task_dicts = [task_dicts]

    for idx, tasks in enumerate(task_dicts):
        # Determine a label for this group
        label = task_names[idx] if task_names and idx < len(task_names) else f"Task Group {idx+1}"
        print(label)
        print("=" * len(label))

        for task_name, metrics in tasks.items():
            print(f"{task_name}:")
            for metric_name, value in metrics.items():
                # Handle numpy arrays
                if isinstance(value, (np.ndarray, list)):
                    arr = np.array(value)
                    if arr.ndim == 1:
                        print(f"  {metric_name}: ", np.array2string(arr, precision=3, separator=', '))
                    elif arr.ndim == 2:
                        print(f"  {metric_name}:")
                        for i, row in enumerate(arr, 1):
                            print(f"    Run {i}: ", np.array2string(row, precision=3, separator=', '))
                    else:
                        print(f"  {metric_name}: array of shape {arr.shape}")
                # Handle scalar numeric values
                elif isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.3f}")
                # Fallback for other types
                else:
                    print(f"  {metric_name}: {value}")
            print()  # blank line between tasks
        print()  # blank line between groups