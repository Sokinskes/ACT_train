"""
AdaStep Training Pipeline for ACT-Plus-Plus
Trains the HorizonPredictor on ACT policy's visual embeddings.

Stage 1: Cluster Analysis & Label Generation (Offline)
Stage 2: Predictor Training (Supervised Learning)
"""

import torch
import numpy as np
import os
import argparse
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictors.adastep import HorizonPredictor, StateClusterAnalyzer
from policy import ACTPolicy
from imitate_episodes import make_policy
from utils import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, help='Path to HDF5 dataset')
    parser.add_argument('--ckpt_dir', required=True, help='ACT policy checkpoint directory')
    parser.add_argument('--ckpt_name', default='policy_last.ckpt', help='Policy checkpoint filename')
    parser.add_argument('--camera_names', nargs='+', default=['top'], help='Camera names')
    parser.add_argument('--k_min', type=int, default=5, help='Minimum horizon')
    parser.add_argument('--k_max', type=int, default=50, help='Maximum horizon')
    parser.add_argument('--num_clusters', type=int, default=10, help='K-Means clusters')
    parser.add_argument('--percentile', type=float, default=50.0, help='Error threshold percentile')
    parser.add_argument('--lambda_param', type=float, default=1.0, help='Safety coefficient')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device')
    return parser.parse_args()


def extract_visual_embeddings(policy, dataset, camera_names, device):
    """
    Extract visual embeddings from ACT policy encoder.
    
    Returns:
        states: [N, latent_dim] - Visual embeddings
        actions: [N, seq_len, action_dim] - Action sequences
    """
    print("\n🔍 Extracting visual embeddings from ACT policy...")
    
    states = []
    actions = []
    
    policy.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing episodes"):
            # Load episode data
            episode = dataset[idx]
            image_data = episode['image']  # [T, C, H, W] or dict
            action_data = episode['action']  # [T, action_dim]
            qpos_data = episode['qpos']  # [T, state_dim]
            
            # Convert to tensors
            if isinstance(image_data, dict):
                images = {k: torch.from_numpy(v).float().to(device) 
                         for k, v in image_data.items()}
            else:
                images = torch.from_numpy(image_data).float().to(device)
            
            qpos = torch.from_numpy(qpos_data).float().to(device)
            
            # Extract features from policy encoder
            if hasattr(policy, 'model') and hasattr(policy.model, 'encoder'):
                # For ACT with CVAE encoder
                latent, _ = policy.model.encoder(qpos, images)  # [T, latent_dim]
            elif hasattr(policy, 'model') and hasattr(policy.model, 'backbone'):
                # For CNN-based policy
                if isinstance(images, dict):
                    feats = []
                    for cam_name in camera_names:
                        feat = policy.model.backbone(images[cam_name])
                        feats.append(feat.flatten(1))
                    latent = torch.cat(feats, dim=1)
                else:
                    latent = policy.model.backbone(images)
                    latent = latent.flatten(1)
            else:
                raise ValueError("Cannot extract features from this policy type")
            
            states.append(latent.cpu().numpy())
            actions.append(action_data)
    
    states = np.concatenate(states, axis=0)  # [N, latent_dim]
    actions = np.concatenate(actions, axis=0)  # [N, seq_len, action_dim]
    
    print(f"✓ Extracted {len(states)} state embeddings")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    
    return states, actions


def generate_labels(states, actions, args):
    """
    Generate horizon labels via cluster analysis and Pareto optimization.
    
    Returns:
        labels: [N] - Optimal horizon for each state
    """
    print("\n🎯 Stage 1: Manifold-Aware Label Generation")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = StateClusterAnalyzer(
        num_clusters=args.num_clusters,
        percentile=args.percentile
    )
    
    # Fit clusters
    analyzer.fit_clusters(states)
    
    # Pareto analysis
    cluster_horizons = analyzer.pareto_analysis(
        states, 
        actions, 
        k_min=args.k_min, 
        k_max=args.k_max,
        lambda_param=args.lambda_param
    )
    
    # Assign labels
    labels = np.array([
        analyzer.get_horizon_for_state(states[i]) 
        for i in tqdm(range(len(states)), desc="Assigning labels")
    ])
    
    # Statistics
    print(f"\n📊 Label Statistics:")
    print(f"  Mean k: {labels.mean():.2f}")
    print(f"  Std k: {labels.std():.2f}")
    print(f"  Range: [{labels.min()}, {labels.max()}]")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Unique values: {len(unique)}")
    
    # Visualize distribution
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=args.k_max - args.k_min + 1, edgecolor='black', alpha=0.7)
    plt.xlabel('Horizon k')
    plt.ylabel('Frequency')
    plt.title('Distribution of Assigned Horizons')
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(args.ckpt_dir, 'horizon_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram to {save_path}")
    
    # Save analyzer
    analyzer_path = os.path.join(args.ckpt_dir, 'cluster_analyzer.pkl')
    analyzer.save(analyzer_path)
    
    return labels


def train_predictor(states, labels, args):
    """
    Train HorizonPredictor via supervised learning.
    """
    print("\n🚀 Stage 2: HorizonPredictor Training")
    print("=" * 60)
    
    device = torch.device(args.device)
    latent_dim = states.shape[1]
    
    # Initialize predictor
    predictor = HorizonPredictor(
        input_dim=latent_dim,
        hidden_dim=256
    ).to(device)
    
    print(f"  Parameters: {predictor.get_num_parameters():,}")
    
    # Prepare data
    states_tensor = torch.from_numpy(states).float()
    labels_normalized = (labels - args.k_min) / (args.k_max - args.k_min)  # Normalize to [0, 1]
    labels_tensor = torch.from_numpy(labels_normalized).float().unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(states_tensor, labels_tensor)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Train
        predictor.train()
        train_loss = 0
        for batch_states, batch_labels in train_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            pred = predictor(batch_states)
            loss = criterion(pred, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        predictor.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_states, batch_labels in val_loader:
                batch_states = batch_states.to(device)
                batch_labels = batch_labels.to(device)
                pred = predictor(batch_states)
                loss = criterion(pred, batch_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': predictor.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'k_min': args.k_min,
                'k_max': args.k_max
            }, os.path.join(args.ckpt_dir, 'horizon_predictor_best.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': predictor.state_dict(),
        'epoch': args.epochs,
        'k_min': args.k_min,
        'k_max': args.k_max
    }, os.path.join(args.ckpt_dir, 'horizon_predictor_last.pth'))
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('HorizonPredictor Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(args.ckpt_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  AdaStep Training Pipeline for ACT-Plus-Plus")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Horizon range: [{args.k_min}, {args.k_max}]")
    print(f"Clusters: {args.num_clusters}")
    print(f"Lambda: {args.lambda_param}")
    
    # Load ACT policy
    print("\n📦 Loading ACT policy...")
    policy_class = 'ACT'
    import pickle
    with open(os.path.join(args.ckpt_dir, 'config.pkl'), 'rb') as f:
        training_config = pickle.load(f)
    policy_config = training_config['policy_config']
    
    policy = make_policy('ACT', policy_config)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    policy.deserialize(torch.load(ckpt_path))
    policy = policy.to(args.device)
    policy.eval()
    print(f"✓ Loaded policy from {ckpt_path}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.dataset_dir}...")
    train_dataloader, val_dataloader, stats, _ = load_data(
        args.dataset_dir,
        lambda n: True,
        camera_names=args.camera_names,
        batch_size_train=8,
        batch_size_val=8,
        chunk_size=policy_config['num_queries']
    )
    
    # Collect full dataset for processing
    dataset = []
    for batch_idx, data in enumerate(train_dataloader):
        dataset.append(data)
    print(f"✓ Loaded {len(dataset)} batches")
    
    # Extract embeddings
    states, actions = extract_visual_embeddings(
        policy, 
        dataset, 
        args.camera_names, 
        args.device
    )
    
    # Generate labels
    labels = generate_labels(states, actions, args)
    
    # Train predictor
    train_predictor(states, labels, args)
    
    print("\n" + "=" * 60)
    print("  ✅ AdaStep Training Complete!")
    print("=" * 60)
    print(f"\nCheckpoint saved to:")
    print(f"  - {os.path.join(args.ckpt_dir, 'horizon_predictor_best.pth')}")
    print(f"  - {os.path.join(args.ckpt_dir, 'cluster_analyzer.pkl')}")

    print("\n🧹 Clean up hardware resources...")
    del states, actions, labels, dataset, policy, train_dataloader
    torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    print("✓ Resources released. Exiting safely.")
    
    import sys
    sys.exit(0)


if __name__ == '__main__':
    main()
