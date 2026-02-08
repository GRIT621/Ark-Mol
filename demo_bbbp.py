from rdkit.Chem import AllChem, DataStructs

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, rdPartialCharges
from collections import Counter

train_df = pd.read_csv(r"./dataset/bbbp/split/train.csv")
valid_df = pd.read_csv(r"./dataset/bbbp/split/valid.csv")
test_df = pd.read_csv(r"./dataset/bbbp/split/test.csv")
sub_r1 ={'c1ccccc1': {'count': 18, 'smile': ['c1ccccc1']}, 'c1ccncc1': {'count': 8, 'smile': ['n1ccccc1']}, 'c1ccsc1': {'count': 6, 'smile': ['s1cccc1']}, 'F': {'count': 2, 'smile': ['F']}, 'Cl': {'count': 13, 'smile': ['Cl']}, 'CO': {'count': 20, 'smile': ['CO']}, 'COC': {'count': 12, 'smile': ['COC']}, 'CN': {'count': 13, 'smile': ['CN']}, 'CS(N)(=O)=O': {'count': 3, 'smile': ['CS(=O)(=O)N']}, 'CN(C)C': {'count': 26, 'smile': ['CN(C)C']}, '[OH]': {'count': 3, 'smile': ['[OH]']}, '[NH2]': {'count': 4, 'smile': ['[NH2]']}, 'O=CO': {'count': 19, 'smile': ['C(=O)[OH]']}, 'CC(C)=O': {'count': 7, 'smile': ['CC(=O)C']}, 'NC=O': {'count': 13, 'smile': ['C(=O)N']}, 'C#N': {'count': 2, 'smile': ['C#N']}, 'CSC': {'count': 13, 'smile': ['CSC']}, 'CC(=O)O': {'count': 7, 'smile': ['CC(=O)O']}, 'Oc1ccccc1': {'count': 9, 'smile': ['c1ccccc1O']}, 'CC(N)=O': {'count': 12, 'smile': ['CC(=O)N']}, 'COC(C)=O': {'count': 4, 'smile': ['COC(=O)C']}, 'Brc1ccccc1': {'count': 1, 'smile': ['Brc1ccccc1']}, 'CCOC(C)=O': {'count': 1, 'smile': ['CCOC(=O)C']}, 'Clc1ccccc1': {'count': 3, 'smile': ['Clc1ccccc1']}, 'Br': {'count': 6, 'smile': ['Br']}, 'CCl': {'count': 7, 'smile': ['CCl']}, 'N=C(N)N': {'count': 11, 'smile': ['N=C(N)N']}, 'COC=O': {'count': 13, 'smile': ['C(=O)OC']}, 'C[NH+](C)C': {'count': 1, 'smile': ['C[NH+](C)C']}, '[Cl]': {'count': 1, 'smile': ['[Cl]']}, 'C=O': {'count': 2, 'smile': ['C=O']}, 'CC#N': {'count': 1, 'smile': ['CC#N']}, 'C[N+](=O)[O-]': {'count': 1, 'smile': ['C[N+](=O)[O-]']}, 'C[N+](C)(C)C': {'count': 2, 'smile': ['C[N+](C)(C)C']}, 'O': {'count': 1, 'smile': ['O']}, 'N': {'count': 1, 'smile': ['N[H]']}, 'C1CCNCC1': {'count': 1, 'smile': ['C1CCNCC1']}, '*N(*)*': {'count': 2, 'smile': ['*N(*)*']}, '*NC(*)=O': {'count': 2, 'smile': ['*C(=O)N*']}, '*N=C(*)N': {'count': 1, 'smile': ['*N=C(N)*']}, '*O': {'count': 2, 'smile': ['*O']}, '*S*': {'count': 1, 'smile': ['*S*']}, '*N': {'count': 2, 'smile': ['*N']}, '*OC(*)=O': {'count': 2, 'smile': ['*C(=O)O*']}, 'C[N+](C)C': {'count': 1, 'smile': ['[N+](C)(C)C']}, '*C(=O)O': {'count': 1, 'smile': ['[*]C(=O)O']}, '*[N+](*)(*)*': {'count': 1, 'smile': ['[*][N+]([*])([*])[*]']}, '*N*': {'count': 1, 'smile': ['[*]N[*]']}, '*Br': {'count': 1, 'smile': ['[*]Br']}, '*C(*)=O': {'count': 1, 'smile': ['[*]C(=O)[*]']}, 'CS': {'count': 1, 'smile': ['CS']}, 'C1CNCCN1': {'count': 1, 'smile': ['N1CCNCC1']}}
pattern_smarts_list = [v['smile'][0] for v in sub_r1.values()]


pattern_mols = [AllChem.MolFromSmarts(s) for s in pattern_smarts_list]

def substructure_features_count(smiles, pattern_mols):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * len(pattern_mols)
    return [len(mol.GetSubstructMatches(p)) if p is not None else 0 for p in pattern_mols]


def aromatic_and_hetero_ring_counts(mol):
    ring_info = mol.GetRingInfo()
    aromatic_rings = 0
    hetero_rings = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_rings += 1
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in [6, 1] for i in ring):
            hetero_rings += 1
    return aromatic_rings, hetero_rings

def aromatic_and_hetero_ring_counts(mol):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    aromatic_count = 0
    hetero_count = 0
    for ring in atom_rings:
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(atom.GetIsAromatic() for atom in atoms):
            aromatic_count += 1
        if any(atom.GetAtomicNum() not in [6, 1] for atom in atoms):  # 非C/H即为杂原子
            hetero_count += 1
    return aromatic_count, hetero_count

def compute_bbb_features(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {}

    features['MolecularWeight'] = Descriptors.MolWt(mol)
    features['LogP'] = Crippen.MolLogP(mol)
    features['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
    features['NumHDonors'] = Lipinski.NumHDonors(mol)
    features['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
    features['FormalCharge'] = AllChem.GetFormalCharge(mol)
    features['NumRotatableBonds'] = Lipinski.NumRotatableBonds(mol)
    features['PgpRiskScore'] = (
        features['TPSA'] / 100.0 +
        features['NumRotatableBonds'] / 10.0 +
        (1 if features['FormalCharge'] != 0 else 0)
    )
    aromatic_rings, hetero_rings = aromatic_and_hetero_ring_counts(mol)
    features['AromaticRings'] = aromatic_rings
    features['HeteroRings'] = hetero_rings
    features['NumRings'] = mol.GetRingInfo().NumRings()
    features['TPSA_per_MW'] = features['TPSA'] / features['MolecularWeight'] if features['MolecularWeight'] else 0
    features['LogP_per_MW'] = features['LogP'] / features['MolecularWeight'] if features['MolecularWeight'] else 0
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    counts = Counter(atom_nums)
    features['C_count'] = counts.get(6, 0)
    features['N_count'] = counts.get(7, 0)
    features['O_count'] = counts.get(8, 0)
    features['F_count'] = counts.get(9, 0)
    features['Cl_count'] = counts.get(17, 0)
    features['Br_count'] = counts.get(35, 0)
    features['S_count'] = counts.get(16, 0)
    features['Tertiary_N_count'] = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetSymbol() == 'N' and atom.GetDegree() == 3 and atom.GetTotalNumHs() == 0
    )
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [abs(float(atom.GetProp('_GasteigerCharge'))) for atom in mol.GetAtoms()]
        features['MaxGasteigerCharge'] = max(charges)
    except:
        features['MaxGasteigerCharge'] = 0.0

    total_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
    features['AromaticAtomRatio'] = aromatic_atoms / total_atoms if total_atoms else 0
    features['HeteroAtomRatio'] = hetero_atoms / total_atoms if total_atoms else 0

    features['BertzCT'] = Descriptors.BertzCT(mol)

    features['LogS'] = (-0.8 * features['LogP']
                        - 0.01 * features['MolecularWeight']
                        + 0.5 * features['NumHDonors']
                        - 0.01 * features['TPSA'])



    return features


def full_features(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    substruct = substructure_features_count(smiles, pattern_mols)
    substruct_np = np.array(substruct, dtype=int)

    bbb_dict = compute_bbb_features(smiles)
    if bbb_dict is None:
        return None
    bbb_np = np.array(list(bbb_dict.values()), dtype=float)
    return np.concatenate([substruct_np,bbb_np])


def extract_features(df):
    feats = []
    feature_names = None
    substruct_names = pattern_smarts_list

    for _, row in df.iterrows():
        f = full_features(row['smiles'])
        if f is not None:
            feats.append(np.append(f, row['p_np']))

            if feature_names is None:
                bbb_feature_names = list(compute_bbb_features(row['smiles']).keys())
                feature_names = substruct_names + bbb_feature_names + ['p_np']

    features_df = pd.DataFrame(feats, columns=feature_names)

    df_filtered = df.loc[features_df.index, df.columns.difference(['p_np'])]

    df_filtered = df_filtered.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    combined_df = pd.concat([df_filtered, features_df], axis=1)
    return combined_df



from autogluon.tabular import TabularPredictor, FeatureMetadata

X_train = extract_features(train_df)
X_valid = extract_features(valid_df)
X_test = extract_features(test_df)

feature_metadata = FeatureMetadata.from_df(X_train)
feature_metadata = feature_metadata.add_special_types({'smiles': ['text']})


predictor = TabularPredictor(label='p_np', eval_metric='roc_auc').fit(train_data=X_train, tuning_data=X_valid)
performance = predictor.evaluate(X_test)

leaderboard = predictor.leaderboard(silent=False)



print(performance)
print(leaderboard)