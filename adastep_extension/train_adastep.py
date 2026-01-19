"""
AdaStep 训练流程
包含状态聚类、帕累托分析、步长预测器训练

使用方法:
1. 先运行一次预训练，生成聚类模型和步长标签
   python train_adastep.py --task task1 --stage pretrain
   
2. 然后运行完整训练
   python train_adastep.py --task task1 --stage train
"""

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG

import os
import pickle
import argparse
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

from training.utils import *
from training.adastep import (
    StateClusterAnalyzer, 
    visualize_clusters, 
    visualize_horizon_distribution
)


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
parser.add_argument('--stage', type=str, default='train', 
                   choices=['pretrain', 'train'],
                   help='pretrain: 聚类分析; train: 完整训练')
args = parser.parse_args()
task = args.task

# 配置
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
os.makedirs(checkpoint_dir, exist_ok=True)

# 设备
device = os.environ['DEVICE']


def pretrain_clustering(train_dataloader, val_dataloader):
    """
    预训练阶段：状态聚类 + 帕累托分析
    生成步长标签用于后续训练
    """
    print("\n" + "="*60)
    print("Stage 1: 预训练 - 状态聚类与帕累托分析")
    print("="*60 + "\n")
    
    # 收集所有状态和动作数据
    print("📊 收集训练数据...")
    all_states = []
    all_actions = []
    all_action_sequences = []
    
    for batch_idx, data in enumerate(train_dataloader):
        image_data, qpos_data, action_data, is_pad = data
        
        # 只取第一个时间步的状态（当前状态）
        states = qpos_data.numpy()  # [batch, state_dim]
        actions = action_data[:, 0].numpy()  # [batch, action_dim]
        action_seqs = action_data.numpy()  # [batch, seq_len, action_dim]
        
        all_states.append(states)
        all_actions.append(actions)
        all_action_sequences.append(action_seqs)
    
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_action_sequences = np.concatenate(all_action_sequences, axis=0)
    
    print(f"✓ 数据收集完成: {all_states.shape[0]} 个样本")
    
    # 创建聚类分析器
    analyzer = StateClusterAnalyzer(
        num_clusters=policy_config['num_clusters'],
        error_threshold=policy_config['error_threshold']
    )
    
    # Step 1: K-Means 聚类
    print("\n🎯 执行 K-Means 聚类...")
    analyzer.fit_clusters(all_states)
    
    # 可视化聚类结果
    labels = analyzer.kmeans.predict(all_states)
    visualize_clusters(
        all_states, 
        labels, 
        os.path.join(checkpoint_dir, 'state_clusters.png')
    )
    
    # Step 2: 帕累托分析
    print("\n📈 执行帕累托分析...")
    cluster_horizons = analyzer.pareto_analysis(
        states=all_states,
        actions=all_actions,
        action_sequences=all_action_sequences,
        k_min=policy_config['k_min'],
        k_max=policy_config['k_max']
    )
    
    print("\n各聚类的最优步长:")
    for cluster_id, horizon in cluster_horizons.items():
        cluster_size = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {horizon} 步 (样本数: {cluster_size})")
    
    # 生成标签
    print("\n🏷️  生成步长标签...")
    horizon_labels = analyzer.get_labels(
        all_states, 
        k_min=policy_config['k_min'],
        k_max=policy_config['k_max']
    )
    
    # 可视化步长分布
    horizons_actual = (horizon_labels * (policy_config['k_max'] - policy_config['k_min']) 
                      + policy_config['k_min'])
    visualize_horizon_distribution(
        horizons_actual.flatten(),
        os.path.join(checkpoint_dir, 'horizon_distribution.png')
    )
    
    # 保存聚类模型
    cluster_path = os.path.join(checkpoint_dir, 'cluster_analyzer.pkl')
    analyzer.save(cluster_path)
    
    # 保存标签
    labels_path = os.path.join(checkpoint_dir, 'horizon_labels.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump({
            'labels': horizon_labels,
            'cluster_ids': labels
        }, f)
    print(f"\n✓ 标签已保存到: {labels_path}")
    
    print("\n" + "="*60)
    print("预训练完成！现在可以运行完整训练")
    print("="*60 + "\n")


def forward_pass(data, policy, horizon_labels_dict=None, batch_idx=None):
    """
    前向传播，支持 AdaStep
    """
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    
    # 如果启用 AdaStep 且有标签
    if policy.use_adastep and horizon_labels_dict is not None:
        # 计算当前批次在全局数据中的索引
        batch_size = qpos_data.shape[0]
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        # 获取对应的标签
        horizon_labels = horizon_labels_dict['labels'][start_idx:end_idx]
        horizon_labels = torch.from_numpy(horizon_labels).float().to(device)
        
        return policy(qpos_data, image_data, action_data, is_pad, horizon_labels)
    else:
        return policy(qpos_data, image_data, action_data, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """保存训练曲线"""
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'训练曲线已保存到 {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, policy_config, horizon_labels_dict=None):
    """
    行为克隆训练（支持 AdaStep）
    """
    print("\n" + "="*60)
    print("Stage 2: 完整训练 - ACT + AdaStep")
    print("="*60 + "\n")
    
    # 加载策略
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)

    # 加载优化器
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    for epoch in range(train_cfg['num_epochs']):
        print(f'\n📍 Epoch {epoch}/{train_cfg["num_epochs"]}')
        
        # 验证
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                # 验证时不使用标签
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        print(f'  Val loss:   {epoch_val_loss:.5f}')
        summary_string = '  '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # 训练
        policy.train()
        optimizer.zero_grad()
        epoch_train_dicts = []
        
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, horizon_labels_dict, batch_idx)
            
            # 反向传播
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_dicts.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        train_history.append(epoch_summary)
        epoch_train_loss = epoch_summary['loss']
        
        print(f'  Train loss: {epoch_train_loss:.5f}')
        summary_string = '  '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # 定期保存
        if epoch % 200 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

    # 保存最终模型
    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    
    # 保存最佳模型
    if best_ckpt_info is not None:
        best_epoch, best_loss, best_state = best_ckpt_info
        ckpt_path = os.path.join(checkpoint_dir, f'policy_best.ckpt')
        torch.save(best_state, ckpt_path)
        print(f"\n✓ 最佳模型: Epoch {best_epoch}, Loss {best_loss:.5f}")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    # 设置随机种子
    set_seed(train_cfg['seed'])
    
    # 数据目录
    data_dir = os.path.join(task_cfg['dataset_dir'], task)
    num_episodes = len(os.listdir(data_dir))

    # 加载数据
    print(f"📁 加载数据: {data_dir}")
    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir, 
        num_episodes, 
        task_cfg['camera_names'],
        train_cfg['batch_size_train'], 
        train_cfg['batch_size_val']
    )
    
    # 保存统计信息
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"✓ 数据统计已保存: {stats_path}\n")

    if args.stage == 'pretrain':
        # 预训练：聚类分析
        pretrain_clustering(train_dataloader, val_dataloader)
        
    elif args.stage == 'train':
        # 完整训练
        if policy_config['use_adastep']:
            # 加载预训练的标签
            labels_path = os.path.join(checkpoint_dir, 'horizon_labels.pkl')
            if not os.path.exists(labels_path):
                print("\n⚠️  警告: 未找到预训练标签！")
                print("请先运行: python train_adastep.py --task {} --stage pretrain".format(task))
                exit(1)
            
            with open(labels_path, 'rb') as f:
                horizon_labels_dict = pickle.load(f)
            print(f"✓ 已加载步长标签: {labels_path}")
        else:
            horizon_labels_dict = None
        
        train_bc(train_dataloader, val_dataloader, policy_config, horizon_labels_dict)
