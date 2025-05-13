from ase.io import read
from mace.calculators import MACECalculator, mace_mp

from heos_relax.evaluation.evaluation_tasks import (
    EnergyForceEval,
    EvalPipelineRunner,
    RelaxRandomSubstitutionTask,
    RelaxTestSetGeometries,
)
from heos_relax.structure_generation import RandomHighEntropyOxideStructureGeneration

from importlib import resources

finetuned_model_path = resources.files("data") / "finetune_mace_heo.model"


pretrained_model = mace_mp("medium", default_dtype="float64",device="cuda", enable_cueq = True)
finetuned_model = MACECalculator(
    model_paths=finetuned_model_path, default_dtype="float64", device="cuda", enable_cueq = True
)


test_set_path = resources.files("data") / "test.xyz"
test_set = read(test_set_path, ":")



energy_force_task = EnergyForceEval(test_set)
relax_task_test_set = RelaxTestSetGeometries(
    test_set[:5], rattle_noise_level=0.001, min_atoms_in_supercell=200, fmax=0.001, N_resamples= 1
)

N_random_structures = 2
random_sub_structures = RandomHighEntropyOxideStructureGeneration(
    space_group=225, composition="Mg1Ni1Cu1Co1Zn1O5"
).generate_structures(N_random_structures)

relax_task_random_substitution = RelaxRandomSubstitutionTask(
    structures=random_sub_structures,
    N_resamples=5,
    rattle_noise_level=0.1,
    fmax=0.05,
)

tasks = [energy_force_task, relax_task_random_substitution]
tasks = [relax_task_random_substitution]
pretrained_model_eval_results = EvalPipelineRunner(tasks).evaluate_model(
    pretrained_model
)
finetuned_model_eval_results = EvalPipelineRunner(tasks).evaluate_model(finetuned_model)


print(pretrained_model_eval_results)

print(finetuned_model_eval_results)
