# PortalCG: Structure-enhanced Deep Meta-learning Predicts Billions of Uncharted Chemical-Protein Interactions on a Genome-scale

![](dark-space-bubble.png)
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

This repository provides script to replicate experiments in PortalCG paper.

# Note to interested users
Our paper put emphasize on proposing new OOD algorithm. We didn't carry out extensive tuning or cherry picking the best one set of model parameters. If you want to use the model for a specific application, we encourage you to follow the described training procedure and carry out model tuning in a swiping style for the best performance.


# Environment setup
Necessary packages to be installed have been specified in <code>environment_setup.txt</code>. Docker file can also be found in the folder<code>environment/</code>

# Data
You can download data and pretrained weights from [here](https://zenodo.org/record/6950921)


__NOTE__: PortalCG has three steps with twice transfer learning. Three large databases are used, Pfam, PDB, ChEMBL, all fairly large. *A full replication from scratch will take a month on a single GPU*. Here, we provide demo with trained weights at final step as well as instructions to replicate from scratch in each step.

## PortalCG training
PortalCG has two major components, (a) STL and (b) OOC-ML with 3 steps in total. Pls run the 3 steps in order.

####  Step1, first transfer learning | (a) STL, step-wise transfer learning
The first step is built on a published work, DISAE (published on JCIM), with replication instructions. In this step, a protein language model will be trained on Pfam with MSA-distilled triplets representation. The whole pfam knowlege will be tranfered to step2.

#### Step2, second transfer learning | (a) STL, step-wise transfer learning 
The second step will train on PDB dataset to predict binding site residue-atom contact map with protein descriptor pretrained in step1 and to be further tuned in this step. 
- To run from scratch: <code>python train_DTI_distance.py --batch_size=128 --eval_at=200 --global_step=40000</code>

 
#### Step3, final DTI prediction | (b) OOC-ML, out-of-cluster meta learning
In this final stage, there are 4 splits of data: OOD-train, iid-dev, OOD-dev,OOD-test. In this step, protein descriptor and chemical descriptor are pretrained in step2.

- To run for a short demo for only 20 steps with trained weights to check OOD-test AUC scores reported in __Table 2__: <code>python train_MAML_4split.py </code>
- To run from scratch: <code>python train_MAML_4split.py --fr_scratch=True --global_step=60000 --global_eval_step=80 </code>

