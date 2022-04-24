# PortalLearning

![](dark-space-bubble.png)
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

This repository provides script to replicate experiments in PortalCG paper


# Environment setup
Necessary packages to be installed have been specified in <code>environment_setup.txt</code>

# Data
You can download data and pretrained weights from [here](https://zenodo.org/record/5701618#.YZHfmmDMKUk)

# PortalCG training demo
On a single GPU, the training on the complete ChEMBL dataset takes a week. Here are the instruction to for demo only with a small subset data. There are two phases of training to be carried out sequentially.  
## Phase 1, STL: stepwise transfer learning
### STL, first step transfer learning
Pls see DISAE (published on JCIM) repository [here](https://github.com/XieResearchGroup/DISAE) 

### STL, second step transfer learning
The first step pretrained weight for protein descriptor will be transferred into the second step. 
run: 

The output for demo is a contact map matrix with average MSE

## Phase 2, OOC: out-of-cluster meta-learning
The pretraiend weight for protein and chemical descriptors from phase1 STL second step will be transferrred into phase2.

run: 

The output for demo


# PortalCG test on DRD3
Fully trained PortalCG weight is shared in the repository. As described in the paper, use trained PortalCG to predict on DRD with wetlab validation. 

run:

The output:

Pls run <code>python undruggable_app.py --cwd=''</code>
