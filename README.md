# 🛰️ DroPTC
<!-- _A Digital Forensics Framework for Drone Log Analysis_ -->

This repository accompanies the research paper:  
> **“Interpretable root cause analysis of drone flight logs”**, submitted to [DFRWS EU, 2026].

It contains:  
- **Source code** (`src/`) for model training, interpretability, and evaluation.  
- **CLI tool** (`src/cli/`) to run the proposed forensic method on new evidence.  
- **Datasets** (`dataset/`, `src/evidence/`) experimental and case study datasets.  
- **Notebooks** (`notebooks/`) for statistical analysis, feature attribution, and visualization.  
- **Experiment outputs** (`experiments/`) where all the experimental results are.  

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/project-name.git
cd project-name
pip install -e .
```

The -e flag installs the package in editable mode, allowing changes without reinstalling. Dependencies are listed in requirements.txt
To install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔬 Reproducing Experimental Results

We provide a single bash script to reproduce all experimental scenarios:
```bash
bash train_classifier.sh
```

This will:
1. Train classifiers under multiple settings.
2. Store results in the `experiments/` folder.
3. Generate trained models that can be reused by the CLI tool.

Analysis of results is available in the `notebooks/` folder:
1. Performance aggregation (mean ± std)
2. Best-performing model analysis
3. Feature attribution heatmaps

## 🛠️ Running the CLI Tool

The forensic tool is provided under the `src/cli/` folder. It supports per-evidence log analysis and outputs timelines + reports. See the dedicated [CLI README](src/cli/README.md) for usage instructions.

## 📂 Repository Structure
```
dropt/
│
├── dataset/            # training and testing datasets
├── experiments/        # experimental results
├── notebooks/          # analysis & visualization notebooks
├── src/                # source code (models, utils, interpretability, CLI, evidence)
│   ├── cli/            # forensic CLI tool
│   ├── evidence/       # forensic case study data (see evidence/README.md)
│   ├── model.py        # model definitions
│   ├── train_classifier.py
│   ├── utils.py
│   └── interpretability.py
│
├── requirements.txt
├── train_classifier.sh
├── README.md           # this file
```

