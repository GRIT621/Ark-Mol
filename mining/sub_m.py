from typing import List, Dict
from mining.base import Base_Mining
from data_utils.dataset_loader import Dataset  # 你的 Dataset 类
from openai import OpenAI


class Substructure_Mining(Base_Mining):
    """基于分子数据的规则生成 Agent"""

    def __init__(self,
                 name: str,
                 dataset: Dataset,
                 model: str = "deepseek-r1",
                 api_key: str = None,
                 api_base: str = None):
        super().__init__(name, model, api_key, api_base)
        self.dataset = dataset

    def _build_prompt(self, samples: List[Dict], subm) -> str:
        """构建用于分子规则生成的提示词"""
        prompt = f"""
        You are a molecular expert and RDKit SMARTS engineer.

        Below is a sample of molecules from the BBBP task:
        {samples}

        Your task:
        Identify and list ten (10) **distinct and independently predictive** molecular substructures that are likely to influence **Blood-Brain Barrier Penetration (BBBP)**.

        Requirements:
        - Each substructure must be chemically meaningful, unique, and non-trivial.
        - Each must be represented as a valid SMARTS string (compatible with RDKit's `Chem.MolFromSmarts()`).
        - Each SMARTS must represent a **new** substructure NOT present in the following exclusion list:
        {subm}

        Format:
        Return the 10 SMARTS substructures only, numbered 1–10. No explanations or additional text.
        e.g. 1. *N=C(*)N  
        2. *S*  
        3. *N(*)*  
        ……

        Strict Rule:
        You must NOT repeat or partially include any of the excluded substructures. Even similar or nested patterns must be avoided.
        """

        return prompt.strip()

    def generate_paginated(self, messages: list, max_tokens=4096, max_pages=5) -> str:
        full_text = ""
        page = 0

        while page < max_pages:
            response = self._call_model(messages)
            choice = response.choices[0]
            if choice.finish_reason != "length":
                return choice.message.content
            content = choice.message.reasoning_content
            full_text += content

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "continue"})
            page += 1

        return full_text

    def generate_substructure(self, samples: List[Dict], subm) -> str:
        prompt = self._build_prompt(samples, subm)

        messages = [{"role": "user", "content": prompt}]

        return self.generate_paginated(messages=messages)

