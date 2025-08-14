# Environment Setup

The following setup notes are written for Ubuntu 24.04 LTS, 
but should be similar for other releases/distributions.


## System Dependencies

Install the required system packages:

```bash
sudo apt install aptitude  nvidia-cuda-toolkit python3-matplotlib python3-pandas build-essential texlive-latex-extra cm-super okular
sudo apt install python3.12-venv
```

## Python Virtual Environment

Create and activate a virtual environment (allowing system packages):

```bash
python3 -m venv pytorch_venv --system-site-packages
source pytorch_venv/bin/activate
```

## Install PyTorch (CUDA 12.4)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Test CUDA Installation

```bash
python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); A = torch.randn(3, 4, device=device); B = torch.randn(4, 2, device=device); C = torch.matmul(A, B); print('PASSED') if device != 'cpu' else print('ERROR')"
```

## Clone and Install Project Dependencies

```bash
git clone https://github.com/decisionintelligence/TFB.git
cd TFB
git checkout ed25d7e
pip3 install -r requirements.txt
cd ..
```

## Additional Dependencies

### For Hyperparameter Tuning:

```bash
pip install hyperopt
```

# Running

The training of models can be done by invoking `hyperparam_tune.py`. 
It will launch a ray session to perform the hyperparameter sweep and automatically save each model's best checkpoint.

Consequently, the checkpoints for models to be evaluated are to be stored in `eval_model_paths.env`.
The env file can be automatically updated with `archive_best_models.py`, which picks the best model from a hyperparameter sweep.

The evaluations can be invoked with their respective start script `eval*_run.sh`.

# Acknowledgement

We would like to thank the authors of the following published code:

https://github.com/decisionintelligence/TFB

https://github.com/Thinklab-SJTU/Crossformer

https://github.com/thuml/Time-Series-Library

# Citation

If you find this dataset or published code useful, please cite our publication via

```
@inproceedings{SRUH2025,
  title         = {Deep Learning-based Time Series Forecasting for Industrial Discrete Process Data},
  publisher     = {IEEE},
  booktitle     = {{8th IEEE Conference on Industrial Cyber-Physical Systems (ICPS)}},
  address       = {Emden, Germany},
  author        = {Sa\ss{}nick, Olaf and Rosenstatter, Thomas and Unterweger, Andreas and Huber, Stefan},
  month         = may,
  year          = 2025,
  doi           = {10.1109/ICPS65515.2025.11087869},
}
```

