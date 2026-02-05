# TxGemma Drug Response Prediction (TxGemma_DRP)

<div align="center">

### Fine-tuning TxGemma for Cancer Drug Response Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**Based on Google's [TxGemma Fine-tuning Cookbook](https://github.com/google-gemini/gemma-cookbook/blob/main/TxGemma/%5BTxGemma%5DFinetune_with_Hugging_Face.ipynb)**

</div>

---

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [HPC Cluster Deployment](#hpc-cluster-deployment)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 About

**TxGemma_DRP** is a fine-tuning pipeline for Google's [TxGemma model](https://huggingface.co/google/txgemma-2b-predict) to predict drug response in cancer cell lines using GDSC (Genomics of Drug Sensitivity in Cancer) data. The project leverages Parameter-Efficient Fine-Tuning (PEFT) with LoRA to adapt the pre-trained TxGemma model for personalized drug response prediction tasks.

This pipeline uses [**ai4clinic**](https://github.com/KatynaSada/ai4clinic), a specialized package for drug response metrics and visualization, to provide comprehensive performance analysis and publication-quality plots.

### What is TxGemma?

TxGemma is a specialized variant of the Gemma language model family, designed specifically for therapeutic and drug discovery applications. It can predict drug responses by understanding molecular structures (SMILES) and cellular contexts.

### Key Applications

- 🧬 **Precision Medicine**: Predict individual patient responses to cancer drugs
- 💊 **Drug Discovery**: Identify promising drug candidates for specific cancer types
- 🔬 **Biomarker Discovery**: Understand drug mechanisms of action
- 📊 **Clinical Decision Support**: Aid oncologists in treatment selection

---

## ✨ Features

- ✅ **Easy-to-use Training Pipeline**: Simplified scripts for model fine-tuning
- ✅ **LoRA Efficient Training**: Parameter-efficient fine-tuning with reduced memory footprint
- ✅ **Multi-GPU Support**: Distributed training with Hugging Face Accelerate
- ✅ **Experiment Tracking**: Integration with Weights & Biases (W&B)
- ✅ **Cross-Validation Support**: Built-in k-fold cross-validation
- ✅ **HPC Cluster Ready**: SLURM job submission scripts included
- ✅ **Comprehensive Metrics**: Pearson/Spearman correlations, RMSE, per-drug analysis using [ai4clinic](https://github.com/KatynaSada/ai4clinic)
- ✅ **Visualization Tools**: Automated generation of performance plots (density scatter, waterfall plots, best-to-worst drug rankings)

---

## 📁 Project Structure

```
TxGemma_DRP/
├── src/
│   ├── finetune_txgemma.py         # Main training script
│   └── compute_final_metrics.py    # Metrics computation and visualization
├── scripts/
│   └── train_txgemma.sh            # Training execution script
├── cluster/
│   └── submit_training.sbatch      # SLURM job submission script
├── environment/
│   └── setup_environment.sh        # Environment setup script
├── data/
│   └── examples/
│       ├── gdsc1_celllines.txt     # Example cell line descriptions
│       └── README.md               # Data format documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # Apache 2.0 License
└── README.md                       # This file
```

---

## 🔧 Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (minimum 16GB VRAM recommended)
- **RAM**: At least 32GB system RAM
- **Storage**: 50GB+ free disk space

### Software Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Python**: 3.9 or higher
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Conda**: Miniconda or Anaconda

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KatynaSada/TxGemma_DRP.git
cd TxGemma_DRP
```

### 2. Automated Environment Setup

The easiest way to set up your environment is using our automated script:

```bash
cd environment
chmod +x setup_environment.sh
./setup_environment.sh
```

This will:
- Create a conda environment named `txgemma_drp`
- Install Python 3.9
- Install CUDA toolkit
- Install PyTorch and all required packages
- Verify the installation

### 3. Manual Installation (Alternative)

If you prefer manual installation:

```bash
# Create conda environment
conda create -n txgemma_drp python=3.9 -y
conda activate txgemma_drp

# Install CUDA toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install ai4clinic for metrics and visualization
pip install ai4clinic
```

**Note**: The [ai4clinic](https://github.com/KatynaSada/ai4clinic) package is required for computing drug response metrics and generating visualization plots.

### 4. Set Up API Keys

```bash
# Set Hugging Face token (required for model access)
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Set Weights & Biases API key (optional, for experiment tracking)
export WANDB_API_KEY="your_wandb_api_key_here"
```

To make these persistent, add them to your `~/.bashrc` or `~/.zshrc`.

---

## ⚡ Quick Start

### Step 1: Prepare Your Data

Organize your data in the following structure:

```
data/
└── GDSC1_drug-blind/
    ├── gdsc1_celllines.txt       # Cell line descriptions
    ├── compound_names.txt         # Drug SMILES and names
    ├── fold1/
    │   ├── train.txt
    │   ├── validate.txt
    │   └── test.txt
    ├── fold2/
    │   └── ...
    └── ...
```

**📄 Example data format**: See [`data/examples/`](data/examples/) for reference files and detailed format documentation.

### Step 2: Configure Training

Edit `scripts/train_txgemma.sh` to set your paths and hyperparameters:

```bash
# Update these variables
DATA_ROOT="/path/to/your/data"
RESULTS_ROOT="/path/to/save/results"
PROJECT="Your_WandB_Project_Name"
ENTITY="your_wandb_entity"
```

### Step 3: Run Training

```bash
# Activate environment
conda activate txgemma_drp

# Navigate to scripts directory
cd scripts

# Make script executable
chmod +x train_txgemma.sh

# Run training
./train_txgemma.sh
```

---

## 📖 Usage

### Training

#### Basic Training

```bash
python src/finetune_txgemma.py \
    --input_folder ./data/GDSC1/ \
    --output_folder ./results/GDSC1/ \
    --fold fold1 \
    --cell_description_file ./data/gdsc1_celllines.txt \
    --train_file_path ./data/GDSC1/fold1/train.txt \
    --train_jsonl_file_path ./data/GDSC1/fold1/train_prompts.jsonl \
    --val_file_path ./data/GDSC1/fold1/validate.txt \
    --val_jsonl_file_path ./data/GDSC1/fold1/validate_prompts.jsonl \
    --test_file_path ./data/GDSC1/fold1/test.txt \
    --test_jsonl_file_path ./data/GDSC1/fold1/test_prompts.jsonl \
    --batch_size 10 \
    --num_train_epochs 40 \
    --learning_rate 5e-6 \
    --wandb_project "TxGemma_DrugResponse" \
    --hf_token $HUGGINGFACE_TOKEN
```

#### Multi-GPU Training

```bash
accelerate launch \
    --num_processes 4 \
    --mixed_precision fp16 \
    src/finetune_txgemma.py \
    [... same arguments as above ...]
```

### Evaluation

After training, compute final metrics across all folds:

```bash
python src/compute_final_metrics.py \
    --input_folder ./data/GDSC1/ \
    --output_folder ./results/GDSC1/ \
    --samples_folders "fold1,fold2,fold3,fold4,fold5" \
    --input_type "mutations" \
    --split_type "drug-blind" \
    --wandb_project "TxGemma_DrugResponse"
```

### HPC Cluster Deployment

For training on SLURM-managed HPC clusters:

1. Edit `cluster/submit_training.sbatch` to match your cluster configuration
2. Submit the job:

```bash
cd cluster
sbatch submit_training.sbatch
```

---

## 📊 Data Format

### Input Files

#### Cell Line Descriptions (`gdsc1_celllines.txt`)
```
Cell_Line_ID,Description
ACH-000001,NIHOVCAR3 is an ovarian carcinoma cell line
ACH-000002,HL60 is an acute promyelocytic leukemia cell line
...
```

#### Training Data (`train.txt`, `validate.txt`, `test.txt`)
```
Cell_Line_ID    SMILES    Sensitivity
ACH-000001    CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F    456.7
ACH-000002    CC(C)CC1=CC=C(C=C1)C(C)C(=O)O    789.2
...
```

#### Compound Names (`compound_names.txt`)
```
Index    SMILES    Drug_Name
1    CC1=CC=C(C=C1)...    Erlotinib
2    CC(C)CC1=CC=C...    Ibuprofen
...
```

---

## ⚙️ Configuration

### Key Hyperparameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `batch_size` | Batch size per GPU | 10 | 8-16 |
| `learning_rate` | Learning rate | 5e-6 | 1e-6 to 1e-5 |
| `lora_r` | LoRA rank | 8 | 4-16 |
| `num_train_epochs` | Training epochs | 40 | 20-50 |
| `max_seq_length` | Max sequence length | 512 | 256-1024 |
| `optimizer` | Optimizer type | adamw_torch | adamw_torch, paged_adamw_8bit |

### Cross-Validation Strategies

- **Random Split**: Random assignment of cell lines and drugs
- **Drug-Blind**: Test on unseen drugs
- **Cell-Blind**: Test on unseen cell lines
- **Cancer-Blind**: Test on unseen cancer types

---

## 📈 Results

The training pipeline generates:

- **Performance Metrics**:
  - Pearson and Spearman correlations
  - Root Mean Square Error (RMSE)
  - Per-drug correlation analysis

- **Visualizations**:
  - Density scatter plots
  - Best-to-worst drug performance plots
  - Waterfall plots of drug correlations

All results are automatically logged to Weights & Biases and saved locally.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{txgemma_drp_2026,
  author = {Sada, Katyna},
  title = {TxGemma Drug Response Prediction},
  year = {2026},
  url = {https://github.com/KatynaSada/TxGemma_DRP}
}
```

Also cite the original TxGemma work:

```bibtex
@article{bommasani2024gemma,
  title={Gemma: Open Models Based on Gemini Research and Technology},
  author={Gemma Team, Google},
  year={2024}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google DeepMind** - For developing and open-sourcing the TxGemma model
- **Hugging Face** - For the transformers library and model hosting
- **GDSC** - For providing the drug sensitivity dataset
- **[ai4clinic](https://github.com/KatynaSada/ai4clinic)** - Our custom package for drug response metrics and visualization
- **Original Tutorial** - Based on [Google Gemini Cookbook - TxGemma Fine-tuning](https://github.com/google-gemini/gemma-cookbook/blob/main/TxGemma/%5BTxGemma%5DFinetune_with_Hugging_Face.ipynb)

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

<div align="center">
Made with ❤️ for advancing precision medicine
</div>
