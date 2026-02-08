# Ark-Mol

---

## Project Structure

```text
Ark-Mol/
│
├── data_utils/
│   └── dataset_loader.py      # Dataset loading and preprocessing
│
├── dataset/
│   └── bbbp/
│       ├── raw/               # Raw BBBP data
│       └── split/             # Train / validation / test splits
│
├── mining/
│   ├── base.py                
│   ├── code_executor.py       # Model execution
│   ├── sub_m.py               # Substructure Mining Module
│   └── task_m.py              # Task Feature Mining Module
│
├── demo_bbbp.py               # BBBP demo
├── execution.py               # Central Coordination and Feekback
├── requirement.txt            # Dependencies

```
We have released the framework pipeline along with a final demo for the BBBP dataset; molecular fragments and physicochemical features for all other datasets will be opened.