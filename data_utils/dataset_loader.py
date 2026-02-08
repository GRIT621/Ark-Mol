import pandas as pd
import random
from typing import List, Dict

class Dataset:
    """分子数据集加载器"""

    def __init__(self, data_path: str, dataset_name:str = "bbbp"):
        self.dataset_name = dataset_name
        self.dataset = pd.read_csv(data_path)
        if self.dataset_name == "bbbp":
            self.score_col = self.dataset.columns[-2]
            self.dataset = self.dataset.sort_values(by=self.score_col, ascending=False)
            self.samples = self.dataset.to_dict('records')

    def sample_batch(self, n: int = 5) -> List[Dict]:
        """随机采样样本"""
        return random.sample(self.samples, min(n, len(self.samples)))

