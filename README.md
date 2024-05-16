# SCSD

Symmetry-Coordinate Structural Decomposition - a tool for crystal structure analysis
Dr. Christopher J. Kingsbury and Prof. Dr. Mathias O. Senge

## Introduction

The scsd (symmetry-coordinate structural decomposition) program is a method of describing the concerted distortions of molecules in crystal structures using the idealised point group symmetry. This can be used for a number of purposes - including structure description, comparison and generation. 

The latest version can be downloaded using the following command:

```bash
conda install -c conda-forge scsd
 ```

This program can be accessed through a notebook interface by importing the underlying module. Many users may find operating through the web interface at https://www.kingsbury.id.au/scsd accessible and suitable. The analysis of known molecules can also be performed with scsd_mercury.py through the Mercury interface, with more information available on the ccdc-opensource repository (https://github.com/ccdc-opensource)

This work can be cited as below:

Kingsbury CJ, Senge MO. Quantifying Near-Symmetric Molecular Distortion Using Symmetry-Coordinate Structural Decomposition. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-6b25s

This citation will be updated after peer review.

While this repository contains tables referring to point groups other than those listed (<i>C</i><sub>2<i>v</i></sub>,<i>C</i><sub>2<i>h</i></sub>,<i>D</i><sub>2<i>d</i></sub>,<i>D</i><sub>2<i>h</i></sub>,<i>D</i><sub>3<i>h</i></sub>,<i>D</i><sub>4<i>h</i></sub>,<i>D</i><sub>6<i>h</i></sub>) these are not verified; check that the atom positions and their sums are in alignment and that the numbers make sense. Use at own risk.

## Examples 

A tutorial notebook is included to demonstrate model generation and analysis, database investigation and conformer generation.

- notebooks/
    - scsd_tutorial.ipynb
    - data
        - bevron.pdb
        - puybaf.pdb
        - pispco.pdb
        - robsons.sd

Examples of integration of these tools with the csd-python-api interface are demonstrated; scsd_mercury.py and scsd_model_mercury.py are intended to be run through the csd-python-api menu in CCDC-Mercury. 

scsd_collection_mercury.py and scsd_single_cl.py contain methods to import data through this interface, and requires a ccdc license.

- scsd_mercury.py
- scsd_model_mercury.py
- scsd_collection_mercury.py

## Dependencies:
- numpy
- scipy
- seaborn
- plotly
- pandas
- networkx
- flask

### Optional
- ipympl
- scikit-learn
- werkzeug
- ccdc

## This work is licensed under THE ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4) 
To view a copy of this license, visit https://directory.fsf.org/wiki/License:ANTI-1.4
Commercial or military use is forbidden.

Please contact Chris Kingsbury (ckingsbury@ccdc.cam.ac.uk) for collaboration or with any questions

## Ongoing issues

To Do List:
- [ ] make 3-fold axis PCAs make sense - there isn't uniformity, and no way to 'normalise' currently
- [ ] comments on the main program aspects
- [ ] "Lacunary" - how to model when a part is missing? FeMoCo is part of this
- [ ] extend sensibly to O<sub>h</sub> ... ongoing work
