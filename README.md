# gw_classification

**Machine-Learning Love: Classifying the Neutron Star Equation of State with Transformers**

This repository provides the code and assets for a neural-network framework that uses the Audio Spectrogram Transformer (AST) architecture to infer neutron-star equations of state from simulated gravitational-wave and electromagnetic time-series data [\[1\]](https://arxiv.org/abs/2210.08382).


## Features

- **AST-based Classification**: Leverages the Audio Spectrogram Transformer to process time-delay cosmography and gravitational-wave spectrograms for equation-of-state inference [\[2\]](https://arxiv.org/abs/2104.01778).
- **End-to-End Pipeline**: Includes data preprocessing, model training, evaluation scripts, and post-processing routines.
- **Simulation-Driven Training**: Trains on synthetic neutron-star observables generated via parameterized spectral fits to relativistic mean-field models.
- **Uncertainty Quantification**: Implements Bayesian model averaging over multiple AST fine-tuning runs to estimate predictive uncertainties.
- **Modular Design**: Easily extendable modules for alternative architectures or new simulation datasets.


## Installation

1. **Clone the Repository**  

git clone https://github.com/G-GoncaIves/gw_classification.git

cd gw_classification

2. **Create a Virtual Environment**  

python3 -m venv venv

source venv/bin/activate


3. **Install Dependencies**  

pip install -r requirements.txt


---

## Quick Start

### 1. Prepare Data  
Generate or download simulated neutron-star spectrograms and place them under 'data/sims/'. 

Run preprocessing:  

python src/preprocess.py --input_dir data/sims/ --output_dir data/preprocessed/


### 2. Train the Model  

Launch training with default parameters:  

python src/train.py
--data_dir data/preprocessed/
--save_dir models/ast_checkpoint
--epochs 50
--batch_size 32


### 3. Evaluate  

Compute classification accuracy and uncertainty estimates:  



---

## Configuration and Experiments

Experiment settings (learning rates, optimizer choices, augmentation flags) are stored as YAML files under 'experiments/'. To reproduce a specific run:  

python src/train.py --config experiments/your_experiment.yaml



---

## Citation

If you use this code, please cite:

> Gonçalves, G. R., et al. “Machine-Learning Love: classifying the equation of state of neutron stars with Transformers.” *Journal of Cosmology and Astroparticle Physics*, 2024. [\[1\]](https://arxiv.org/abs/2210.08382)
