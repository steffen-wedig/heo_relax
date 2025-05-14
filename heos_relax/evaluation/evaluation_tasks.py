import math
from abc import ABC, abstractmethod
from ase.calculators.calculator import BaseCalculator
import numpy as np
from ase import Atoms
from ase.build import make_supercell
from mace.calculators import MACECalculator, mace_mp
from tqdm import tqdm

from heos_relax.evaluation.eval_utils import relax_material_FIRE, relax_material_LBFGS


class EvalTask(ABC):
    @abstractmethod
    def evaluate(self, model: MACECalculator, model_name: str):
        pass


class EnergyForceEval(EvalTask):
    """
    An Eval Task to calculate Energy and force RMSE on specific dataset splits
    """

    def __init__(self, materials: list[Atoms]):
        self.materials = materials

    def evaluate(self, model: MACECalculator, model_name: str):
        sum_sq_energy = 0.0
        n_structures = len(self.materials)

        sum_sq_forces = 0.0
        n_force_components = 0  # total number of force components across all materials

        for mat in tqdm(self.materials):
            # true DFT values
            e_true = mat.info["DFT_energy"]
            f_true = mat.arrays["DFT_forces"]

            # model predictions
            mat.calc = model
            e_pred = mat.get_potential_energy()
            f_pred = mat.get_forces()

            n_atoms = len(mat)

            # accumulate energy squared error
            sum_sq_energy += ((e_pred - e_true) / n_atoms) ** 2

            # accumulate force squared error and count
            diff = f_pred - f_true
            sum_sq_forces += np.sum(diff**2)
            n_force_components += diff.size

        # compute RMSEs
        rmse_energy = np.sqrt(sum_sq_energy / n_structures) * 1000
        rmse_forces = np.sqrt(sum_sq_forces / n_force_components) * 1000

        return {
            "rmse_energy_meV/atom": rmse_energy,
            "rmse_forces_meV/A/atom": rmse_forces,
        }


class RelaxTestSetGeometries(EvalTask):
    """Relaxes perturbed geometries from the test set."""

    def __init__(
        self,
        materials: list[Atoms],
        N_resamples: int,
        rattle_noise_level: float = 0.001,
        min_atoms_in_supercell: int = 200,
        fmax: float = 0.001,
    ):
        self.materials = materials

        self.N_resamples = N_resamples
        self.rattle_noise_level = rattle_noise_level
        self.min_atoms_in_supercell = min_atoms_in_supercell
        self.fmax = fmax

    def evaluate(
        self,
        model: MACECalculator,
        model_name: str
    ):
        volume_changes = []
        frac_coord_errors = []

        random_number_generator = np.random.default_rng(0)

        for material in self.materials:
            N_atoms = len(material)

            max_reference_force_norm = np.max(
                np.linalg.norm(material.arrays["DFT_forces"], axis=1)
            )

            if max_reference_force_norm > self.fmax:
                print("Reference Structure is not at equilibrium")
                continue

            supercell_dim = math.ceil((self.min_atoms_in_supercell / N_atoms)**(1/3))

            for _ in range(self.N_resamples):
                mat = make_supercell(material, P=np.identity(3) * supercell_dim)
                reference_volume = mat.get_volume()
                reference_frac_coords = mat.get_scaled_positions()

                mat.rattle(stdev=self.rattle_noise_level, rng=random_number_generator)
                relax_material_FIRE(mat, model)

                # Calculate cell vector loss
                relaxed_volume = mat.get_volume()

                relative_volume_change = (
                    np.abs(reference_volume - relaxed_volume) / reference_volume
                )
                volume_changes.append(relative_volume_change)

                relaxed_frac_coords = mat.get_scaled_positions()
                frac_coord_error = np.sum(
                    (relaxed_frac_coords - reference_frac_coords) ** 2
                )  # This is not a metric in the mathematical sense, so I am not sure how useful this is for benchmarking the performance of a model.
                frac_coord_errors.append(frac_coord_error)

        return {
            "mean_fractional_coordinate_error": np.mean(np.array(frac_coord_errors)),
            "mean_relative_volume_change": np.mean(np.array(volume_changes)),
        }


class RelaxRandomSubstitutionTask(EvalTask):
    def __init__(
        self,
        structures: list[Atoms],
        N_resamples: int,
        reference_calculator : BaseCalculator, 
        rattle_noise_level: float = 0.001,
        fmax: float = 0.001,
        N_max_optimiziation_steps: int = 100,
    ):
        self.structures = structures
        self.N_resamples = N_resamples
        self.reference_calc = reference_calculator
        self.rattle_noise_level = rattle_noise_level
        self.fmax = fmax
        self.N_max_optimization_steps = N_max_optimiziation_steps

    def evaluate(
        self,
        model: MACECalculator,
        model_name: str
    ):
        random_number_generator = np.random.default_rng(0)
        
        runs_converged = np.zeros(shape=(len(self.structures), self.N_resamples), dtype = bool)
        final_energies = np.zeros(shape=(len(self.structures), self.N_resamples))
        steps_taken = np.zeros(shape=(len(self.structures), self.N_resamples))

        reference_energies =  np.zeros(shape=(len(self.structures), self.N_resamples))

        for structure_idx, atoms in enumerate(self.structures):
            for resampling_idx in range(self.N_resamples):
                structure_to_relax = atoms.copy()
                structure_to_relax.rattle(
                    stdev=self.rattle_noise_level, rng=random_number_generator
                )

                trajectory_file = f"{structure_to_relax.get_chemical_formula()}_{resampling_idx}_{model_name}.traj"

                try:
                    converged_flag, N_steps= relax_material_LBFGS(
                        material=structure_to_relax,
                        mace_calc=model,
                        ftol=self.fmax,
                        max_N_optimization_steps=self.N_max_optimization_steps,
                        trajectory_file= trajectory_file
                    )
                except:
                    continue

                final_energies[structure_idx, resampling_idx] = (
                    structure_to_relax.get_potential_energy()
                )

                
                reference_energies[structure_idx, resampling_idx]= self.reference_calc.get_potential_energy(structure_to_relax)

                runs_converged[structure_idx, resampling_idx] = converged_flag
                steps_taken[structure_idx, resampling_idx] = N_steps


        mean_steps_to_convergence = np.mean(steps_taken, axis=0, where=runs_converged)

        ratio_runs_converged = np.sum(runs_converged) / runs_converged.size

        results = {
            "mean_steps_to_convergence": mean_steps_to_convergence,
            "ratio_runs_converged": ratio_runs_converged,
            "final_energies": final_energies,
            "reference_energies" : reference_energies
        }

        return results


class EvalPipelineRunner:
    def __init__(self, tasks: list[EvalTask]):
        self.tasks = tasks

    def evaluate_model(self, model: MACECalculator, model_name : str):
        # Runs all tasks stored with the given model

        result_dict = {}
        for task in self.tasks:
            try:
                task_name = task.__class__.__name__
                assert task_name not in result_dict
                result_dict[task_name] = task.evaluate(model, model_name)
            except Exception as e:
                print(f"{e} in Task {task_name}. Continuing.")
                continue

        return result_dict
