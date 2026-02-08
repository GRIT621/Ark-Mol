from typing import List, Dict
from mining.base import Base_Mining
from data_utils.dataset_loader import Dataset
from openai import OpenAI


class Task_Mining(Base_Mining):
    """基于分子数据的规则生成 Agent"""

    def __init__(self,
                 name: str,
                 dataset: Dataset,
                 model: str = "deepseek-r1",
                 api_key: str = None,
                 api_base: str = None):
        super().__init__(name, model, api_key, api_base)
        self.dataset = dataset

    def _build_prompt(self, task_description: str, samples: List[Dict]) -> str:
            prompt = f"""
    You are a molecular chemistry expert and Python developer. 

    Task Context:
    {task_description}

    Input Data Samples:
    {samples}

    Based on the task and chemical principles, identify 10 distinct molecular features or physicochemical properties likely to influence the outcome.

    Requirements:
    1. For each feature, you must provide a unique **Feature Name** and the exact **RDKit Python Code Snippet** to calculate it.
    2. The code snippet must assume a RDKit molecule object named `mol` is already defined.
    3. Use standard RDKit modules: `Descriptors`, `Crippen`, `rdMolDescriptors`, `Lipinski`, or `fragments`.
    4. Return ONLY a numbered list in the format: `Index. FeatureName | CodeSnippet`. No explanations.

    Example Output Format:
    1. MolWeight | Descriptors.MolWt(mol)
    2. LogP | Crippen.MolLogP(mol)
    3. TPSA | rdMolDescriptors.CalcTPSA(mol)
    4. TPSA_Density | rdMolDescriptors.CalcTPSA(mol) / Descriptors.MolWt(mol)
    5. H_Bond_Donors | Lipinski.NumHDonors(mol)
    ...
    """
            return prompt.strip()


    def generate_taskfeature(self, samples: List[Dict]) -> str:
        prompt = self._build_prompt(samples)
        try:
            return self._call_model(prompt)
        except Exception as e:
            return f"[Error] {str(e)}"

