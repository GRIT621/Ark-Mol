import os
os.environ["RDLOGLEVEL"] = "ERROR"
from dotenv import load_dotenv
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import re
import pandas as pd
from rdkit import Chem

load_dotenv()

import time
import logging
import argparse
import copy
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import requests
from rdkit import RDLogger
import os
from typing import Dict, List
from mining.task_m import  *
from mining.sub_m import  *
# from agents.code_agent import  *
from mining.code_executor import *

lg = RDLogger.logger()
# lg.setLevel(RDLogger.ERROR)
def setup_logging(log_path):

    os.makedirs(log_path, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("test_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_path, log_filename)
    log_format = '%(asctime)s [%(levelname)s] %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Log saved at：%s", os.path.abspath(log_filepath))
# logging.disable(logging.CRITICAL)



def analyze_shap_interactions3(
    shap_values,
    shap_inter_values,
    feature_names= None,
    main_thresh=0.01,
    interact_thresh=0.01,
    max_val_thresh=0.01,
    nonzero_ratio_thresh=0.05
):
    shap_values = np.array(shap_values)
    shap_inter_values = np.array(shap_inter_values)
    n_samples, n_feat = shap_values.shape

    if feature_names is None:
        feature_names = [str(i) for i in range(n_feat)]

    main_effect = np.mean(np.abs(shap_values), axis=0)
    max_main_vals = np.max(np.abs(shap_values), axis=0)
    nonzero_ratio = np.mean(np.abs(shap_values) > 1e-5, axis=0)

    main_from_inter = np.mean(np.abs(shap_inter_values[:, range(n_feat), range(n_feat)]), axis=0)
    total_interaction = np.mean(np.abs(shap_inter_values), axis=0).sum(axis=1) - main_from_inter

    df = pd.DataFrame({
        'SHAP_main': main_effect,
        'SHAP_interaction': total_interaction,
        'max_main_SHAP': max_main_vals,
        'nonzero_ratio': nonzero_ratio
    }, index=feature_names)

    def classify(row):
        main = row['SHAP_main']
        inter = row['SHAP_interaction']
        max_shap = row['max_main_SHAP']
        nz_ratio = row['nonzero_ratio']

        if main >= main_thresh and inter >= interact_thresh:
            return "Retain (Key Driver)"
        elif main >= main_thresh and inter < interact_thresh:
            return "Retain (Independent Main)"
        elif main < main_thresh and inter >= interact_thresh:
            return "Retain (Synergistic Feature)"
        elif max_shap > max_val_thresh or nz_ratio > nonzero_ratio_thresh:
            return "Retain (Local/Sparse Importance)"
        else:
            return "Candidates for Removal/Replacement"

    df['Strategy_Recommendation'] = df.apply(classify, axis=1)

    return df.sort_values(by='SHAP_main', ascending=True)



class CentralControlAgent:
    def __init__(self):

        self.subm = {}
        self.subt = {}


    def _parse_structure_str(self, structure_str: str) -> list:
        structures = []
        for line in structure_str.strip().split('\n'):
            match = re.match(
                r'^\s*\d+[.)]\s*`?([^`\s]+)`?\s*$',
                line.strip()
            )
            if match and match.group(1):
                structures.append(match.group(1))
        return structures

    def add_subm(self, structure_str: str):
        structures = self._parse_structure_str(structure_str)
        for smile in structures:
            try:
                mol = Chem.MolFromSmarts(smile)
                if mol is None:
                    continue
                canonical = Chem.MolToSmarts(mol)
                if canonical in self.subm:
                    self.subm[canonical]["count"] += 1
                    if smile not in self.subm[canonical]['smile']:
                        self.subm[canonical]['smile'].append(smile)
                else:
                    self.subm[canonical] = {"count": 1, "smile": [smile]}
            except:
                pass

    def add_subt(self, task_str: str):
        lines = task_str.strip().split('\n')
        for line in lines:
            try:
                content = line.split('.', 1)[-1]
                name, code = content.split('|')
                self.subt[name.strip()] = code.strip()
            except:
                continue

    def filter_all_features(self, shap_values, shap_interaction_values):
        subm_keys = list(self.subm.keys())
        subt_keys = list(self.subt.keys())
        all_feature_names = subm_keys + subt_keys
        n_all = len(all_feature_names)
        current_shap = shap_values[:, :n_all]
        current_inter = shap_interaction_values[:, :n_all, :n_all]

        df = analyze_shap_interactions3(
            shap_values=current_shap,
            shap_inter_values=current_inter,
            feature_names=all_feature_names,
            main_thresh=0.01,
            interact_thresh=0.01,
            max_val_thresh=0.01,
            nonzero_ratio_thresh=0.05
        )
        to_remove = df[df['Strategy_Recommendation'] == 'Candidates for Removal/Replacement'].index.tolist()

        removed_subm = 0
        removed_subt = 0

        for k in to_remove:
            if k in self.subm:
                self.subm.pop(k, None)
                removed_subm += 1
            elif k in self.subt:
                self.subt.pop(k, None)
                removed_subt += 1

        logging.info(f"Pruning Complete: Removed {removed_subm} substructures and {removed_subt} task features.")
        logging.info(f"Remaining: {len(self.subm)} subms, {len(self.subt)} subts.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"")
    parser.add_argument("--model", type=str, default="deepseek-r1")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--test_rounds", type=int, default=50)
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--log_path",
                        type=str,
                        default=r"")
    args = parser.parse_args()
    setup_logging(args.log_path)

    top10_badcase = None

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found:{args.data_path}")


    try:
        dataset = Dataset(args.data_path)

        submolecular_generator = Substructure_Mining(
            name="SubstructureMining",
            dataset=dataset,
            model="deepseek-r1",
            api_key=args.api_key,
            api_base=args.api_base,
        )

        taskfeature_generator = Task_Mining(
            name="TaskMining",
            dataset=dataset,
            model="deepseek-r1",
            api_key=args.api_key,
            api_base=args.api_base
        )
        CentralControl = CentralControlAgent()

        executor = CodeExecutor()


    except Exception as e:
        logging.exception(f"Initialization failed:{str(e)}")
        return

    for i in range(args.test_rounds):

        if i == 0:
            samples = dataset.sample_batch(5)
        else:
            samples = top10_badcase
        submolecular = submolecular_generator.generate_substructure(samples, CentralControl.subm)
        while not submolecular:
            time.sleep(5)
            submolecular = submolecular_generator.generate_substructure(samples, CentralControl.subm)
        logging.info(f"\n[Turn {i + 1} Substructure:")
        logging.info(submolecular)

        CentralControl.add_subm(submolecular)

        sub_t = taskfeature_generator.generate_taskfeature(samples)
        while not sub_t:
            time.sleep(5)
            sub_t = taskfeature_generator.generate_taskfeature(samples)

            logging.info(f"\n[Turn {i + 1} Task Feature:")
            logging.info(sub_t)

        CentralControl.add_subt(sub_t)

        AUC, top10_badcase, shap_values, shap_interaction_values = executor.execute_code(args.data_path,CentralControl.subm,CentralControl.subt)
        logging.info(f"AUC: {AUC:.5f}")
        if i % 10 == 0:
            prev_subm = CentralControl.subm.copy()
            prev_keys = set(prev_subm.keys())
            prev_auc = AUC

            CentralControl.filter_subm(shap_values, shap_interaction_values)
            new_keys = set(CentralControl.subm.keys())
            removed_keys = prev_keys - new_keys

            new_auc, _, new_shap_values, new_shap_inter_values = executor.execute_code(args.data_path,
                                                                                       CentralControl.subm,CentralControl.subt)
            logging.info(f"[Iteration {i + 1}] Post-pruning AUC: {new_auc:.4f}")

            if new_auc < prev_auc:

                CentralControl.subm = prev_subm
            else:
                logging.info(f"[Iteration {i + 1}] Optimization successful. AUC improved or stabilized.")

        logging.info("\n=== Feature Mining & Optimization Completed ===")


if __name__ == "__main__":
    main()