# GGL-ETA-Score

This code compute features for both the SYBYL-GGL and ECIF-GGL models. The folder `src` contains the main source code. The code `get_ggl_features.py` can be used to generate features for a given protein-ligand dataset. 

## Package Requirement
- NumPy
- SciPy
- Pandas
- Biopandas
- RdKit

Run the following command to install the necessary packages and create an conda environmet from the provided `ggl-score-env.yml` file. 
```shell
conda env create -f ggl-score-env.yml
```

## Simple Example
Assume we want to genrate the features for the PDBbind v2016 general set for both SYBYL GGL and ECIF GGL with exponential kernel type and parameters $\kappa=2.5$ and $\tau=1.5$ which is the index 84 of the `kernels.csv` file in the `utils` folder. Assume also the structes of the dataset are in the directory `../PDBbind_v2016_general_Set` and we wish to save the features in the directory `../Features`.

```shell
# Generate the SYBYL GGL features for the PDBbind v2016 general set
python get_ggl_features.py -k 84 -c 12.0 -m 'SYBYL' -f '../csv_data_file/PDBbindv2016_GeneralSet.csv' -dd '../PDBbind_v2016_general_set' -fd '../Features'

# Generate the ECIF GGL features for the PDBbind v2016 general set
python get_ggl_features.py -k 84 -c 12.0 -m 'ECIF' -f '../csv_data_file/PDBbindv2016_GeneralSet.csv' -dd '../PDBbind_v2016_general_set' -fd '../Features'
```
