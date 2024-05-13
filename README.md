CURRENTLY UNDER CONSTRUCTION
---
# SCSD
Symmetry-Coordinate Structural Decomposition - a tool for crystal structure analysis

The scsd (symmetry-coordinate structural decomposition) program is a method of describing the concerted distortions of molecules in crystal structures using the idealised point group symmetry. This can be used for a number of purposes - including structure description, comparison and generation. 

The latest version can be downloaded using the following command:

> pip install scsd

This program can be accessed through a notebook interface by importing the underlying module. Many users may find operating through the web interface at https://www.kingsbury.id.au/scsd accessible and suitable. The analysis of known molecules can also be performed with scsd_mercury.py through the Mercury interface, with more information available on the ccdc-opensource repository (https://github.com/ccdc-opensource)

This work can be cited as below:

Kingsbury CJ, Senge MO. Quantifying Near-Symmetric Molecular Distortion Using Symmetry-Coordinate Structural Decomposition. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-6b25s

This citation will be updated after peer review.

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

# This work is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ 
# Commercial or military use is forbidden.

Please contact Chris Kingsbury (ckingsbury@ccdc.cam.ac.uk) for collaboration or with any questions

### Ongoing issues

To Do List:
- make 3-fold axis PCAs make sense - there isn't uniformity, and no way to 'normalise' currently
- comments on the main program aspects

- some way of generating and preserving custom models for users - done 27/9/21
- Figure an easy webserver upload for the new data/version - done 15/3/22
- update scsd_direct and easy_database_gen to the new program version - done 15/5/22
- remove the * imports - done 13/5/24
- "Lacunary" - how to model when a part is missing? The FeMoco is part of this, I guess
- Paths - turn into relative paths at some point - done 11/5/24
