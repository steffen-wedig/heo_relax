# Finetuning ML Potentials for relaxing high entropy oxides

Mini-project for application to Dunia Internship - Steffen Wedig

In this mini-project, we finetune a machine learning potential (MACE-MP-0) for relaxation of high entropy oxides (HEOs). Finding stable/ low-energy geometries is important for

- characterising the properties of the bulk material (eg. phonon accuracy, defect energies, )
- determining surface properties (adsorption energies, catalytic activities)

We take three steps to tackle this task:

1. Curate a small dataset of high entropy oxides.
2. Finetune a MACE-MP-0 model on energies and forces
3. Run relaxation starting from randomly substituted crystals

## Dataset Curation

We collect our training dataset from the LeMaterials Dataset, which is a combination of QMOD, Alexandria, and the Materials Project.

Simple Definition of HEOs:

- Contains Oxygen,
- Five or more elements,
- All non-oxygen elements are metals

Also filter out data samples out which do not contain forces

Could possibly add more filters:

- Configurational entropy higher then 1.5 R
- Equimolarity of constituent cations
- Remove expensive cations (noble metals e.g. platinum)

Visualize elemental distribution, UMAP projections, show some exemplary structures.

## Finetuning

Finetune MACE-MP-0 using the naive finetuning protocol from the mace torch package.

We forego replay finetuning, because replaying the original training data drastically increases the computational cost of finetuning, and for this mini-project, forgetting is deemed not to be important. Further, the LeMaterials dataset contains relevant mp materials anyways, which would be contained in the replay data as well.

## Relaxation/Evaluation

Compare the energy/force RMSEs for finetuned model

In-distribution eval task: Take the DFT relaxed structures, rattle and relax with MLIP and check the root-mean-square distance (RMSD)/energy difference to the reference.

Out-of-distribution eval task: Start from random substituted parent crystals and relax, and check which model reaches lower energies. However, without performing further DFT, we can't really validate which model  

Compare rate of convergence for both models.

