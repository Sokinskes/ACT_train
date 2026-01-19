
import sys
import os
import torch
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add path to root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adastep_module import HorizonPredictor, StateClusterAnalyzer
from data.robomimic_loader import create_robomimic_dataloaders
from validation.offline_validator import OfflineValidator

# Re-implement stage functions to support lambda_param

def run_clustering_and_analysis(data_loader, save_dir, config, lambda_param):
    print(f"\n📊 Collecting data for lambda={lambda_param}...")
    all_states = []
    all_action_seqs = []
    
    for images, qpos, actions, is_pad in data_loader:
        all_states.append(qpos.numpy())
        all_action_seqs.append(actions.numpy())
    
    all_states = np.concatenate(all_states, axis=0)
    all_action_seqs = np.concatenate(all_action_seqs, axis=0)
    
    analyzer = StateClusterAnalyzer(
        num_clusters=config['num_clusters'],
        error_threshold=config['error_threshold']
    )
    
    analyzer.fit_clusters(all_states)
    
    # Pareto analysis with specific lambda
    horizon_map = analyzer.pareto_analysis(
        all_states, 
        all_action_seqs,
        k_min=config['k_min'],
        k_max=config['k_max'],
        lambda_param=lambda_param
    )
    
    os.makedirs(save_dir, exist_ok=True)
    analyzer.save(os.path.join(save_dir, 'cluster_analyzer.pkl'))
    
    labels = analyzer.get_labels(all_states, config['k_min'], config['k_max'])
    np.save(os.path.join(save_dir, 'horizon_labels.npy'), labels)
    
    # Analyze distribution statistics
    k_values = list(horizon_map.values())
    stats = {
        'lambda': lambda_param,
        'mean_k': np.mean(k_values),
        'min_k': np.min(k_values),
        'max_k': np.max(k_values),
        'efficiency': np.mean(k_values) / 1.0  # Relative to k=1
    }
    return analyzer, labels, stats

def train_mlp(train_loader, val_loader, analyzer, labels, save_dir, config):
    device = config['device']
    predictor = HorizonPredictor(input_dim=config['state_dim'], hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    
    print(f"🎓 Training MLP...")
    for epoch in range(config['num_epochs']):
        predictor.train()
        label_idx = 0
        for _, qpos, _, _ in train_loader:
            qpos = qpos.to(device)
            batch_size = qpos.shape[0]
            batch_labels = labels[label_idx:label_idx+batch_size]
            batch_labels = torch.from_numpy(batch_labels).float().to(device)
            label_idx += batch_size
            
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predictor(qpos), batch_labels)
            loss.backward()
            optimizer.step()
            
    torch.save(predictor.state_dict(), os.path.join(save_dir, 'model.pth'))
    return predictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./experiments/sensitivity_results')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    lambdas = [0.5, 1.0, 2.0]
    results = []
    
    # Load data once
    train_loader, val_loader, stats = create_robomimic_dataloaders(
        args.data_path, max_episodes=20, batch_size_train=32, batch_size_val=32
    )
    
    config = {
        'k_min': 5, 'k_max': 50,
        'num_clusters': 10,  # Improved granularity
        'error_threshold': 0.5, # Median
        'state_dim': stats['qpos_dim'],
        'num_epochs': args.epochs,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    for lam in lambdas:
        print(f"\n\n>>> Running Sensitivity Analysis for Lambda = {lam} <<<")
        exp_dir = os.path.join(args.output_dir, f"lambda_{lam}")
        
        # 1. Clustering & Analysis
        analyzer, labels, stats = run_clustering_and_analysis(
            train_loader, exp_dir, config, lambda_param=lam
        )
        
        # 2. Train (Simplified)
        predictor = train_mlp(train_loader, val_loader, analyzer, labels, exp_dir, config)
        
        # 3. Validation (Accuracy Only for speed)
        validator = OfflineValidator(
            predictor, analyzer, val_loader, k_min=5, k_max=50, device=config['device']
        )
        acc_res = validator.validation_1_accuracy(exp_dir)
        stats['accuracy'] = acc_res['accuracy']
        
        results.append(stats)
        
    # Summary
    df = pd.DataFrame(results)
    print("\n\n=== Sensitivity Analysis Summary ===")
    print(df)
    df.to_csv(os.path.join(args.output_dir, 'sensitivity_summary.csv'))

if __name__ == '__main__':
    main()
