# One-off runner to evaluate lambda=3.0 and append results to sensitivity_results
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_sensitivity_experiment import run_clustering_and_analysis, train_mlp
from validation.offline_validator import OfflineValidator
from data.robomimic_loader import create_robomimic_dataloaders
import numpy as np
import pandas as pd

DATA_PATH = '../robomimic_data/square/mh/low_dim_v15.hdf5'
OUT_DIR = './sensitivity_results'
LAM = 3.0
EPOCHS = 5

os.makedirs(OUT_DIR, exist_ok=True)
train_loader, val_loader, stats = create_robomimic_dataloaders(DATA_PATH, max_episodes=20, batch_size_train=32, batch_size_val=32)
config = {
    'k_min': 5, 'k_max': 50,
    'num_clusters': 10,
    'error_threshold': 0.5,
    'state_dim': stats['qpos_dim'],
    'num_epochs': EPOCHS,
    'device': 'cuda' if False else 'cpu'
}

exp_dir = os.path.join(OUT_DIR, f'lambda_{LAM}')
_analyzer, _labels, stats = run_clustering_and_analysis(train_loader, exp_dir, config, lambda_param=LAM)
_predictor = train_mlp(train_loader, val_loader, _analyzer, _labels, exp_dir, config)
validator = OfflineValidator(_predictor, _analyzer, val_loader, k_min=5, k_max=50, device=config['device'])
acc_res = validator.validation_1_accuracy(exp_dir)
stats['accuracy'] = acc_res['accuracy']
# append to CSV
csv_path = os.path.join(OUT_DIR, 'sensitivity_summary.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame()

df = df.append(pd.Series(stats, name=len(df)), ignore_index=False)
df.to_csv(csv_path, index=False)
print('Done. Appended lambda=', LAM, '->', stats)
