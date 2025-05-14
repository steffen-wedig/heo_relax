from pathlib import Path

from ase.io import read
from mace.calculators import MACECalculator, mace_mp

from heos_relax.evaluation.eval_utils import pretty_print_tasks
from heos_relax.evaluation.evaluation_tasks import (
    EnergyForceEval,
    EvalPipelineRunner,
    RelaxRandomSubstitutionTask,
)
from heos_relax.structure_generation import RandomHighEntropyOxideStructureGeneration

import argparse

parser = argparse.ArgumentParser(description="Load the LeMaterials Training Data")
parser.add_argument(
    "--dir",
    dest="dir",
    type=str,
    required=True,
    help="Path to the data directory"
)
args = parser.parse_args()
data_dir = Path(args.dir)
finetuned_model_path = data_dir / "finetune_mace_heo.model"


pretrained_model = mace_mp(
    "medium", default_dtype="float64", device="cuda", enable_cueq=True
)

finetuned_model = MACECalculator(
    model_paths=finetuned_model_path,
    default_dtype="float64",
    device="cuda",
    enable_cueq=True,
)

# In the abscence of DFT, we use a different MACE Model (OMAT) as reference calaculator, which generally does quite well with crystaline materials and small displacements (e.g. very good phono spectra)
reference_calc = MACECalculator(
    model_paths=data_dir / "mace-omat-0-medium.model",
    device="cuda",
    enable_cueq=True,
    default_dtype="float64",
)


test_set_path = data_dir / "test.xyz"
test_set = read(test_set_path, ":")

energy_force_task = EnergyForceEval(test_set)


N_random_structures = 2
random_sub_structures = RandomHighEntropyOxideStructureGeneration(
    composition="Mg1Ni1Cu1Co1Zn1O5"
).generate_structures(N_random_structures)

relax_task_random_substitution = RelaxRandomSubstitutionTask(
    structures=random_sub_structures,
    reference_calculator=reference_calc,
    N_resamples=5,
    rattle_noise_level=0.1,
    fmax=0.05,
)

tasks = [energy_force_task, relax_task_random_substitution]


pretrained_model_eval_results = EvalPipelineRunner(tasks).evaluate_model(
    pretrained_model, "pretrained"
)
finetuned_model_eval_results = EvalPipelineRunner(tasks).evaluate_model(
    finetuned_model, "finetuned"
)


pretty_print_tasks(
    [pretrained_model_eval_results, finetuned_model_eval_results],
    ["pretrained", "finetuned"],
)
