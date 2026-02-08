from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, rdPartialCharges
from collections import Counter
import shap
from autogluon.tabular import TabularPredictor

class CodeExecutor:
    def __init__(self):
        self.substructure_patterns = {}

    def compute_task_features(self, smiles, taskfeature):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        features = {}

        for feat_name, code_snippet in taskfeature.items():
            try:
                features[feat_name] = eval(code_snippet)
            except Exception as e:
                features[feat_name] = 0.0
        return features


    def substructure_features_count(self, smiles, pattern_mols):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * len(pattern_mols)
        return [len(mol.GetSubstructMatches(p)) if p is not None else 0 for p in pattern_mols]

    def full_features(self, smiles, pattern_mols, taskfeature):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        substruct_np = np.array(self.substructure_features_count(smiles, pattern_mols), dtype=int)

        task_dict = self.compute_task_features(smiles, taskfeature)
        if task_dict is None:
            return None
        task_np = np.array(list(task_dict.values()), dtype=float)

        return np.concatenate([substruct_np, task_np])

    def extract_features(self, df, pattern_mols,taskfeature):
        feats, labels = [], []
        for _, row in df.iterrows():
            f = self.full_features(row['smiles'],pattern_mols,taskfeature)
            if f is not None:
                feats.append(f)
                labels.append(row['p_np'])
        return np.array(feats), np.array(labels)
    def execute_code(self, data_name=None, substructures=None, taskfeature = None):
        if substructures is not None:
            self.substructure_patterns = [v['smile'][0] for v in substructures.values()]
        train_data = pd.read_csv(r'.\dataset\bbbp\split\train.csv')
        valid_data = pd.read_csv(r'.\dataset\bbbp\split\valid.csv')

        pattern_mols = [Chem.MolFromSmarts(s) for s in substructures]

        X_train, y_train = self.extract_features(train_data, pattern_mols,taskfeature)
        X_valid, y_valid = self.extract_features(valid_data, pattern_mols,taskfeature)


        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train

        valid_df = pd.DataFrame(X_valid)
        valid_df['target'] = y_valid

        predictor = TabularPredictor(
            label='target',
            eval_metric='roc_auc',
            path='ag_models/'
        ).fit(
            train_data=train_df,
            tuning_data=valid_df,
            time_limit=300,
            presets='best_quality'
        )

        y_pred_proba = predictor.predict_proba(valid_df.drop(columns=['target']))
        y_pred_proba_val = y_pred_proba.iloc[:, 1].values
        AUC = roc_auc_score(y_valid, y_pred_proba_val)
        print(f"AutoGluon Best Model Valid ROC-AUC: {AUC:.5f}")

        y_pred_proba_train = predictor.predict_proba(train_df.drop(columns=['target'])).iloc[:, 1].values
        errors = abs(y_train - y_pred_proba_train)
        top10_idx = np.argpartition(errors, -10)[-10:]
        top10_badcase = train_data.iloc[top10_idx]
        best_model_name = predictor.get_model_best()

        try:
            explainer = shap.Explainer(predictor.predict, train_df.drop(columns=['target']).sample(100))
            shap_values = explainer(train_df.drop(columns=['target']).sample(100)).values
            shap_interaction_values = None
        except Exception as e:
            print(f"SHAP calculation failed, using Feature Importance instead. Error: {e}")
            importance_df = predictor.feature_importance(train_df)
            shap_values = importance_df['importance'].values  # 模拟 SHAP 格式返回
            shap_interaction_values = None

        return AUC, top10_badcase, shap_values, shap_interaction_values






