#!/usr/bin/env python

"""
Introduction:
    GGL-Score: Geometric Graph Learning Score

Author:
    Masud Rana (masud.rana@uky.edu)

Date last modified:
    Nov 25, 2022

"""

import numpy as np
import pandas as pd
import os
from os import listdir
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2


class KernelFunction:

    def __init__(self, kernel_type='exponential_kernel',
                 kappa=2.0, tau=1.0):
        self.kernel_type = kernel_type
        self.kappa = kappa
        self.tau = tau

        self.kernel_function = self.build_kernel_function(kernel_type)

    def build_kernel_function(self, kernel_type):
        if kernel_type[0] in ['E', 'e']:
            return self.exponential_kernel
        elif kernel_type[0] in ['L', 'l']:
            return self.lorentz_kernel

    def exponential_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii

        return np.exp(-(d/eta)**self.kappa)

    def lorentz_kernel(self, d, vdw_radii):
        eta = self.tau*vdw_radii

        return 1/(1+(d/eta)**self.kappa)


class SYBYL_GGL:

    protein_atom_types_df = pd.read_csv(
        '../utils/protein_atom_types.csv')

    ligand_atom_types_df = pd.read_csv(
        '../utils/ligand_SYBYL_atom_types.csv')

    protein_atom_types = protein_atom_types_df['AtomType'].tolist()
    protein_atom_radii = protein_atom_types_df['Radius'].tolist()

    ligand_atom_types = ligand_atom_types_df['AtomType'].tolist()
    ligand_atom_radii = ligand_atom_types_df['Radius'].tolist()

    protein_ligand_atom_types = [
        i[0]+"-"+i[1] for i in product(protein_atom_types, ligand_atom_types)]

    def __init__(self, Kernel, cutoff):

        self.Kernel = Kernel
        self.cutoff = cutoff

        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()

    def get_pairwise_atom_type_radii(self):

        protein_atom_radii_dict = {a: r for (a, r) in zip(
            self.protein_atom_types, self.protein_atom_radii)}

        ligand_atom_radii_dict = {a: r for (a, r) in zip(
            self.ligand_atom_types, self.ligand_atom_radii)}

        pairwise_atom_type_radii = {i[0]+"-"+i[1]: protein_atom_radii_dict[i[0]] +
                                    ligand_atom_radii_dict[i[1]] for i in product(self.protein_atom_types, self.ligand_atom_types)}

        return pairwise_atom_type_radii

    def mol2_to_df(self, mol2_file):
        df_mol2 = PandasMol2().read_mol2(mol2_file).df
        df = pd.DataFrame(data={'ATOM_INDEX': df_mol2['atom_id'],
                                'ATOM_ELEMENT': df_mol2['atom_type'],
                                'X': df_mol2['x'],
                                'Y': df_mol2['y'],
                                'Z': df_mol2['z']})

        if len(set(df["ATOM_ELEMENT"]) - set(self.ligand_atom_types)) > 0:
            print(
                "WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
        return(df)

    def pdb_to_df(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)
        ppdb_all_df = ppdb.df['ATOM']
        ppdb_df = ppdb_all_df[ppdb_all_df['atom_name'].isin(
            self.protein_atom_types)]
        atom_index = ppdb_df['atom_number']
        atom_element = ppdb_df['atom_name']
        x, y, z = ppdb_df['x_coord'], ppdb_df['y_coord'], ppdb_df['z_coord']
        df = pd.DataFrame.from_dict({'ATOM_INDEX': atom_index, 'ATOM_ELEMENT': atom_element,
                                     'X': x, 'Y': y, 'Z': z})

        return df

    def get_mwcg_rigidity(self, protein_file, ligand_file):
        '''
            Adapted from ECIF package
        '''
        protein = self.pdb_to_df(protein_file)
        ligand = self.mol2_to_df(ligand_file)

        # select protein atoms in a cubic with a size of cutoff from ligand
        for i in ["X", "Y", "Z"]:
            protein = protein[protein[i] < float(ligand[i].max())+self.cutoff]
            protein = protein[protein[i] > float(ligand[i].min())-self.cutoff]

        atom_pairs = list(
            product(protein["ATOM_ELEMENT"], ligand["ATOM_ELEMENT"]))
        atom_pairs = [x[0]+"-"+x[1] for x in atom_pairs]
        pairwise_radii = [self.pairwise_atom_type_radii[x]
                          for x in atom_pairs]
        pairwise_radii = np.asarray(pairwise_radii)

        pairwise_mwcg = pd.DataFrame(atom_pairs, columns=["ATOM_PAIR"])
        distances = cdist(protein[["X", "Y", "Z"]],
                          ligand[["X", "Y", "Z"]], metric="euclidean")
        pairwise_radii = pairwise_radii.reshape(
            distances.shape[0], distances.shape[1])
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        distances = distances.ravel()
        mwcg_distances = mwcg_distances.ravel()
        mwcg_distances = pd.DataFrame(
            data={"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances], axis=1)
        pairwise_mwcg = pairwise_mwcg[pairwise_mwcg["DISTANCE"] <= self.cutoff].reset_index(
            drop=True)

        return pairwise_mwcg

    def get_ggl_score(self, protein_file, ligand_file):
        features = ['COUNTS', 'SUM', 'MEAN', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ligand_file)
        mwcg_temp_grouped = pairwise_mwcg.groupby('ATOM_PAIR')
        mwcg_temp_grouped.agg(['sum', 'mean', 'std', 'min', 'max'])
        mwcg_temp = mwcg_temp_grouped.size().to_frame(name='COUNTS')
        mwcg_temp = (mwcg_temp
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'sum'}).rename(columns={'MWCG_DISTANCE': 'SUM'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'mean'}).rename(columns={'MWCG_DISTANCE': 'MEAN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'std'}).rename(columns={'MWCG_DISTANCE': 'STD'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'min'}).rename(columns={'MWCG_DISTANCE': 'MIN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'max'}).rename(columns={'MWCG_DISTANCE': 'MAX'}))
                     )
        mwcg_columns = {'ATOM_PAIR': self.protein_ligand_atom_types}
        for _f in features:
            mwcg_columns[_f] = np.zeros(len(self.protein_ligand_atom_types))
        ggl_score = pd.DataFrame(data=mwcg_columns)
        ggl_score = ggl_score.set_index('ATOM_PAIR').add(
            mwcg_temp, fill_value=0).reindex(self.protein_ligand_atom_types).reset_index()

        return ggl_score


class ECIF_GGL:
    ECIF_ProteinAtoms = ['C;4;1;3;0;0', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1',
                         'C;4;3;0;0;0', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                         'C;5;3;0;0;0', 'C;6;3;0;0;0', 'N;3;1;2;0;0', 'N;3;2;0;1;1',
                         'N;3;2;1;0;0', 'N;3;2;1;1;1', 'N;3;3;0;0;1', 'N;4;1;2;0;0',
                         'N;4;1;3;0;0', 'N;4;2;1;0;0', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                         'S;2;1;1;0;0', 'S;2;2;0;0;0']

    ECIF_LigandAtoms = ['As;1;1;0;0;0', 'As;4;4;0;0;0', 'B;1;1;0;0;0', 'B;2;2;0;0;0',
                        'B;3;2;1;0;0', 'B;3;3;0;0;0', 'B;3;3;0;0;1', 'B;3;4;0;0;0',
                        'B;3;4;0;0;1', 'B;4;4;0;0;0', 'B;5;4;0;0;0', 'B;5;5;0;0;0',
                        'Be;4;4;0;0;0', 'Br;1;1;0;0;0', 'Br;4;4;0;0;0', 'C;1;1;0;0;0',
                        'C;2;2;0;0;0', 'C;2;2;0;0;1', 'C;3;2;0;0;0', 'C;3;2;0;0;1',
                        'C;3;2;0;1;1', 'C;3;2;1;0;1', 'C;3;2;2;0;0', 'C;3;3;0;0;0',
                        'C;3;3;0;0;1', 'C;3;3;0;1;1', 'C;3;3;1;0;1', 'C;4;1;1;0;0',
                        'C;4;1;2;0;0', 'C;4;1;3;0;0', 'C;4;2;0;0;0', 'C;4;2;0;0;1',
                        'C;4;2;1;0;0', 'C;4;2;1;0;1', 'C;4;2;1;1;1', 'C;4;2;2;0;0',
                        'C;4;2;2;0;1', 'C;4;3;0;0;0', 'C;4;3;0;0;1', 'C;4;3;0;1;1',
                        'C;4;3;1;0;0', 'C;4;3;1;0;1', 'C;4;3;1;1;1', 'C;4;4;0;0;0',
                        'C;4;4;0;0;1', 'C;4;4;0;1;1', 'C;5;2;0;0;0', 'C;5;2;1;0;1',
                        'C;5;2;2;0;0', 'C;5;2;3;0;0', 'C;5;3;0;0;0', 'C;5;3;0;0;1',
                        'C;5;3;0;1;1', 'C;5;3;1;0;1', 'C;5;3;2;0;0', 'C;5;3;2;0;1',
                        'C;5;4;1;0;1', 'C;6;3;0;0;0', 'C;6;6;0;0;0', 'Cl;1;1;0;0;0',
                        'Cl;2;1;1;0;0', 'Cl;2;2;0;0;1', 'Cl;3;3;0;0;1', 'Cu;2;2;0;0;1',
                        'Cu;4;4;0;0;1', 'F;1;1;0;0;0', 'F;2;1;1;0;0', 'Fe;4;4;0;0;1',
                        'Fe;5;5;0;0;1', 'Fe;6;4;2;0;1', 'Fe;6;5;1;0;1', 'Fe;6;6;0;0;1',
                        'Fe;9;9;0;0;1', 'Hg;1;1;0;0;0', 'I;1;1;0;0;0', 'Ir;3;4;0;0;1',
                        'Mg;4;4;0;0;1', 'N;2;1;1;0;0', 'N;2;2;0;0;0', 'N;2;2;0;0;1',
                        'N;3;1;0;0;0', 'N;3;1;1;0;0', 'N;3;1;2;0;0', 'N;3;2;0;0;0',
                        'N;3;2;0;0;1', 'N;3;2;0;1;1', 'N;3;2;1;0;0', 'N;3;2;1;0;1',
                        'N;3;2;1;1;1', 'N;3;2;2;0;1', 'N;3;3;0;0;0', 'N;3;3;0;0;1',
                        'N;3;3;0;1;1', 'N;4;1;2;0;0', 'N;4;1;3;0;0', 'N;4;2;0;0;0',
                        'N;4;2;0;0;1', 'N;4;2;1;0;0', 'N;4;2;1;0;1', 'N;4;2;2;0;0',
                        'N;4;2;2;0;1', 'N;4;3;0;0;0', 'N;4;3;0;0;1', 'N;4;3;1;0;0',
                        'N;4;3;1;0;1', 'N;4;4;0;0;0', 'N;4;4;0;0;1', 'N;5;2;0;0;0',
                        'N;5;2;1;0;0', 'N;5;3;0;0;0', 'N;5;3;0;1;1', 'N;5;4;0;0;0',
                        'O;1;1;0;0;0', 'O;1;1;1;0;0', 'O;1;2;0;0;1', 'O;2;0;2;0;0',
                        'O;2;1;0;0;0', 'O;2;1;1;0;0', 'O;2;2;0;0;0', 'O;2;2;0;0;1',
                        'O;2;2;0;1;1', 'O;3;1;0;0;0', 'O;3;1;1;0;0', 'O;3;1;2;0;0',
                        'O;3;2;0;0;1', 'O;3;2;1;0;0', 'O;3;2;1;0;1', 'Os;8;8;0;0;1',
                        'P;4;3;0;0;0', 'P;4;4;0;0;0', 'P;5;3;0;0;0', 'P;5;3;1;0;0',
                        'P;5;4;0;0;0', 'P;5;4;0;0;1', 'P;6;3;1;0;0', 'P;6;4;0;0;0',
                        'P;6;4;0;0;1', 'P;7;4;0;0;0', 'Pt;0;2;0;0;1', 'Pt;3;3;0;0;1',
                        'Re;8;8;0;0;1', 'Rh;1;3;0;0;1', 'Rh;9;9;0;0;1', 'Ru;6;6;0;0;1',
                        'Ru;7;7;0;0;1', 'Ru;8;8;0;0;1', 'Ru;9;9;0;0;1', 'S;2;1;0;0;0',
                        'S;2;1;1;0;0', 'S;2;2;0;0;0', 'S;2;2;0;0;1', 'S;3;2;0;0;1',
                        'S;3;3;0;0;0', 'S;3;3;0;0;1', 'S;4;3;0;0;0', 'S;5;3;0;0;0',
                        'S;5;4;0;0;0', 'S;6;3;1;0;0', 'S;6;4;0;0;0', 'S;6;4;0;0;1',
                        'S;7;4;0;0;0', 'Sb;7;4;0;0;0', 'Se;1;1;0;0;0', 'Se;2;2;0;0;0',
                        'Se;6;1;5;0;0', 'Si;4;4;0;0;0', 'V;8;5;0;0;1']

    PossibleECIF = [i[0]+"-"+i[1]
                    for i in product(ECIF_ProteinAtoms, ECIF_LigandAtoms)]

    atom_type_radii = {
        'As': 1.85,
        'B': 0.85,
        'Be': 1.53,
        'Br': 1.85,
        'C': 1.7,
        'Cl': 1.75,
        'Cu': 1.28,
        'F': 1.47,
        'Fe': 1.26,
        'Hg': 1.5,
        'I': 1.98,
        'Ir': 2.0,
        'Mg': 1.73,
        'N': 1.55,
        'O': 1.52,
        'Os': 2.0,
        'P': 1.8,
        'Pt': 1.75,
        'Re': 2.05,
        'Rh': 2.0,
        'Ru': 2.05,
        'S': 1.8,
        'Sb': 2.06,
        'Se': 1.9,
        'Si': 2.1,
        'V': 1.34
    }

    Atom_Keys = pd.read_csv(
        "../utils//PDB_Atom_Keys.csv", sep=",")

    def __init__(self, Kernel, cutoff):
        self.Kernel = Kernel
        self.cutoff = cutoff

    def GetAtomType(self, atom):
        # This function takes an atom in a molecule and returns its type as defined for ECIF

        AtomType = [atom.GetSymbol(),
                    str(atom.GetExplicitValence()),
                    str(len([x.GetSymbol()
                             for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                    str(len([x.GetSymbol()
                             for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                    str(int(atom.GetIsAromatic())),
                    str(int(atom.IsInRing())),
                    ]

        return(";".join(AtomType))

    def LoadSDFasDF(self, SDF):
        # This function takes an SDF for a ligand as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF

        m = Chem.MolFromMolFile(SDF, sanitize=False)
        m.UpdatePropertyCache(strict=False)

        ECIF_atoms = []

        for atom in m.GetAtoms():
            if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
                entry = [int(atom.GetIdx())]
                entry.append(self.GetAtomType(atom))
                pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
                entry.append(float("{0:.4f}".format(pos.x)))
                entry.append(float("{0:.4f}".format(pos.y)))
                entry.append(float("{0:.4f}".format(pos.z)))
                ECIF_atoms.append(entry)

        df = pd.DataFrame(ECIF_atoms)
        df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]
        if len(set(df["ECIF_ATOM_TYPE"]) - set(self.ECIF_LigandAtoms)) > 0:
            print(
                "WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
            print(set(df["ECIF_ATOM_TYPE"]) - set(self.ECIF_LigandAtoms))
        return(df)

    def LoadPDBasDF(self, PDB):
        # This function takes a PDB for a protein as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF

        ECIF_atoms = []

        f = open(PDB)
        for i in f:
            if i[:4] == "ATOM":
                # Include only non-hydrogen atoms
                if (
                    len(i[12:16].replace(" ", "")) < 4 and 
                    i[12:16].replace(" ", "")[0] != "H") or (len(i[12:16].replace(" ", "")) == 4 and 
                    i[12:16].replace(" ", "")[1] != "H" and 
                    i[12:16].replace(" ", "")[0] != "H"
                    ):

                    ECIF_atoms.append([int(i[6:11]),
                                       i[17:20]+"-"+i[12:16].replace(" ", ""),
                                       float(i[30:38]),
                                       float(i[38:46]),
                                       float(i[46:54])
                                       ])

        f.close()

        df = pd.DataFrame(ECIF_atoms, columns=[
                          "ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
        df = df.merge(self.Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
            ["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
        if list(df["ECIF_ATOM_TYPE"].isna()).count(True) > 0:
            print(
                "WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
        return(df)

    def get_mwcg_rigidity(self, protein_file, ligand_file):
        '''
            Adapted from ECIF package
        '''
        protein = self.LoadPDBasDF(protein_file)
        ligand = self.LoadSDFasDF(ligand_file)

        # select protein atoms in a cubic with a size of cutoff from ligand
        for i in ["X", "Y", "Z"]:
            protein = protein[protein[i] < float(ligand[i].max())+self.cutoff]
            protein = protein[protein[i] > float(ligand[i].min())-self.cutoff]

        atom_pairs = list(
            product(protein["ECIF_ATOM_TYPE"], ligand["ECIF_ATOM_TYPE"]))

        atom_pairs = [x[0]+"-"+x[1] for x in atom_pairs]
        atom_pairs = pd.DataFrame(atom_pairs, columns=["ATOM_PAIR"])
        atom_pairs["ELEMENTS_PAIR"] = [x.split(
            "-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in atom_pairs["ATOM_PAIR"]]
        atom_pairs["PAIR_RADII"] = [self.atom_type_radii[x.split(
            "-")[0]] + self.atom_type_radii[x.split("-")[1]] for x in atom_pairs["ELEMENTS_PAIR"]]

        pairwise_radii = atom_pairs['PAIR_RADII'].values

        pairwise_mwcg = atom_pairs.drop(
            ["ELEMENTS_PAIR", "PAIR_RADII"], axis=1)

        distances = cdist(protein[["X", "Y", "Z"]],
                          ligand[["X", "Y", "Z"]], metric="euclidean")
        pairwise_radii = pairwise_radii.reshape(
            distances.shape[0], distances.shape[1])
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        distances = distances.ravel()
        mwcg_distances = mwcg_distances.ravel()
        mwcg_distances = pd.DataFrame(
            data={"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances], axis=1)
        pairwise_mwcg = pairwise_mwcg[pairwise_mwcg["DISTANCE"] <= self.cutoff].reset_index(
            drop=True)

        return pairwise_mwcg

    def get_ggl_score(self, protein_file, ligand_file):
        features = ['COUNTS', 'SUM', 'MEAN', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ligand_file)
        mwcg_temp_grouped = pairwise_mwcg.groupby('ATOM_PAIR')
        mwcg_temp_grouped.agg(['sum', 'mean', 'std', 'min', 'max'])
        mwcg_temp = mwcg_temp_grouped.size().to_frame(name='COUNTS')
        mwcg_temp = (mwcg_temp
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'sum'}).rename(columns={'MWCG_DISTANCE': 'SUM'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'mean'}).rename(columns={'MWCG_DISTANCE': 'MEAN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'std'}).rename(columns={'MWCG_DISTANCE': 'STD'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'min'}).rename(columns={'MWCG_DISTANCE': 'MIN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'max'}).rename(columns={'MWCG_DISTANCE': 'MAX'}))
                     )

        mwcg_columns = {'ATOM_PAIR': self.PossibleECIF}
        for _f in features:
            mwcg_columns[_f] = np.zeros(len(self.PossibleECIF))
        ggl_score = pd.DataFrame(data=mwcg_columns)
        ggl_score = ggl_score.set_index('ATOM_PAIR').add(
            mwcg_temp, fill_value=0).reindex(self.PossibleECIF).reset_index()

        return ggl_score
