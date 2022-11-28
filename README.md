# GGL-ETA-Score

This code compute features for both the SYBYL-GGL and ECIF-GGL models. The folder `src` contains the main source code. The code `get_ggl_features.py` can be used to generate features for a given protein-ligand dataset. 

## Package Requirement
- NumPy
- SciPy
- Pandas
- Biopandas
- RdKit
To install the necessary packages from the `conda-forge` channel and create an conda environmet from the provided `ggl-score-env.yml` file, run the following command
```shell
conda env create -f ggl-score-env.yml
```
