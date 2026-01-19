"""
Robomimic数据加载器
专门处理Robomimic Square (Nut Assembly)任务的HDF5数据

数据格式:
- 状态: robot proprio (14-dim for bimanual, 7-dim for single arm)
- 图像: RGB cameras (multiple views)
- 动作: end-effector pose delta or joint velocity
- 轨迹长度: ~400-600 steps per episode

下载地址:
https://robomimic.github.io/docs/datasets/robomimic_v0.1.html
选择: Square (ph) - Proficient Human demonstrations
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import os
from PIL import Image


class RobomimicSquareDataset(Dataset):
    """
    Robomimic Square任务数据集
    
    兼容ACT的数据格式要求:
    - 返回 (image, qpos, action, is_pad)
    """
    
    def __init__(self, 
                 hdf5_path: str,
                 camera_names: List[str] = ['agentview_image', 'robot0_eye_in_hand_image'],
                 max_episodes: int = 50,  # ACT典型值：50条轨迹
                 chunk_size: int = 100,    # ACT的num_queries
                 image_size: Tuple[int, int] = (480, 640)):
        """
        Args:
            hdf5_path: Robomimic数据集路径
            camera_names: 相机名称列表
            max_episodes: 最大加载轨迹数（ACT数据高效，50条足够）
            chunk_size: 动作序列长度（对应ACT的chunk）
            image_size: 图像尺寸 (H, W)
        """
        self.hdf5_path = hdf5_path
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.image_size = image_size
        
        # 加载数据
        print(f"📂 加载Robomimic数据: {hdf5_path}")
        self.episodes = self._load_episodes(max_episodes)
        
        # 计算统计量（用于归一化）
        self.stats = self._compute_stats()
        
        print(f"✓ 加载完成: {len(self.episodes)} 条轨迹")
        print(f"  状态维度: {self.stats['qpos_dim']}")
        print(f"  动作维度: {self.stats['action_dim']}")
        print(f"  平均长度: {self.stats['avg_length']:.1f} 步")
    
    def _load_episodes(self, max_episodes: int) -> List[Dict]:
        """加载轨迹数据"""
        episodes = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demos = list(f['data'].keys())[:max_episodes]
            
            for demo_name in demos:
                demo = f[f'data/{demo_name}']
                
                # 读取状态（robot proprio）
                if 'obs/robot0_eef_pos' in demo:
                    # 使用末端执行器位姿
                    eef_pos = demo['obs/robot0_eef_pos'][()]
                    eef_quat = demo['obs/robot0_eef_quat'][()]
                    qpos = np.concatenate([eef_pos, eef_quat], axis=-1)
                elif 'obs/robot0_joint_pos' in demo:
                    # 使用关节角度
                    qpos = demo['obs/robot0_joint_pos'][()]
                else:
                    raise ValueError("未找到有效的状态数据！")
                
                # 读取动作
                actions = demo['actions'][()]
                
                # 读取图像
                images = {}
                for cam_name in self.camera_names:
                    key = f'obs/{cam_name}'
                    if key in demo:
                        images[cam_name] = demo[key][()]
                
                # 检查长度一致性
                length = len(qpos)
                assert len(actions) == length, "状态和动作长度不匹配！"
                
                episodes.append({
                    'qpos': qpos,
                    'actions': actions,
                    'images': images,
                    'length': length
                })
        
        return episodes
    
    def _compute_stats(self) -> Dict:
        """计算数据统计量"""
        all_qpos = []
        all_actions = []
        lengths = []
        
        for ep in self.episodes:
            all_qpos.append(ep['qpos'])
            all_actions.append(ep['actions'])
            lengths.append(ep['length'])
        
        all_qpos = np.concatenate(all_qpos, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        stats = {
            'qpos_mean': all_qpos.mean(axis=0),
            'qpos_std': all_qpos.std(axis=0) + 1e-6,
            'action_mean': all_actions.mean(axis=0),
            'action_std': all_actions.std(axis=0) + 1e-6,
            'qpos_dim': all_qpos.shape[1],
            'action_dim': all_actions.shape[1],
            'avg_length': np.mean(lengths)
        }
        
        return stats
    
    def __len__(self) -> int:
        """数据集总样本数"""
        total = 0
        for ep in self.episodes:
            # 每条轨迹可以生成 (length - chunk_size) 个样本
            total += max(1, ep['length'] - self.chunk_size + 1)
        return total
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        获取一个样本
        
        Returns:
            image: [num_cameras, 3, H, W]
            qpos: [qpos_dim]
            actions: [chunk_size, action_dim]
            is_pad: [chunk_size] - padding mask
        """
        # 找到对应的episode和时间步
        ep_idx = 0
        local_idx = idx
        
        for ep_idx, ep in enumerate(self.episodes):
            ep_samples = max(1, ep['length'] - self.chunk_size + 1)
            if local_idx < ep_samples:
                break
            local_idx -= ep_samples
        
        episode = self.episodes[ep_idx]
        t = local_idx
        
        # 提取状态
        qpos = episode['qpos'][t].astype(np.float32)
        qpos = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        
        # 提取动作序列
        action_seq = []
        is_pad = []
        
        for i in range(self.chunk_size):
            if t + i < episode['length']:
                action = episode['actions'][t + i].astype(np.float32)
                action = (action - self.stats['action_mean']) / self.stats['action_std']
                action_seq.append(action)
                is_pad.append(False)
            else:
                # padding
                action_seq.append(np.zeros_like(episode['actions'][0]))
                is_pad.append(True)
        
        actions = np.stack(action_seq, axis=0)
        is_pad = np.array(is_pad)
        
        # 提取图像
        images = []
        for cam_name in self.camera_names:
            if cam_name in episode['images']:
                img = episode['images'][cam_name][t]
                
                # 调整尺寸
                img = Image.fromarray(img)
                img = img.resize((self.image_size[1], self.image_size[0]))
                img = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
                img = img.astype(np.float32) / 255.0
                images.append(img)
        
        images = np.stack(images, axis=0) if images else np.zeros((1, 3, *self.image_size))
        
        return (
            torch.from_numpy(images).float(),
            torch.from_numpy(qpos).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(is_pad)
        )


def create_robomimic_dataloaders(
    hdf5_path: str,
    batch_size_train: int = 8,
    batch_size_val: int = 8,
    val_ratio: float = 0.1,
    max_episodes: int = 50,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器
    
    Args:
        hdf5_path: Robomimic数据集路径
        batch_size_train: 训练批次大小
        batch_size_val: 验证批次大小
        val_ratio: 验证集比例
        max_episodes: 最大轨迹数
    
    Returns:
        train_loader, val_loader, stats
    """
    # 加载完整数据集
    full_dataset = RobomimicSquareDataset(hdf5_path, max_episodes=max_episodes, **kwargs)
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ 数据集划分: 训练 {train_size} / 验证 {val_size}")
    
    return train_loader, val_loader, full_dataset.stats


# ===== 辅助函数 =====

def download_robomimic_dataset(task: str = 'square', 
                               dataset_type: str = 'ph',
                               save_dir: str = './robomimic_data'):
    """
    下载Robomimic数据集的辅助函数
    
    Args:
        task: 任务名称 ('square', 'transport', 'can', 'lift', 'tool_hang')
        dataset_type: 数据类型 ('ph' - Proficient Human, 'mg' - Multi-Human)
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Robomimic v0.1 数据集URL
    base_url = "http://downloads.cs.stanford.edu/downloads/rt_benchmark/robomimic_v0.1"
    filename = f"{task}_{dataset_type}.hdf5"
    url = f"{base_url}/{filename}"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        print(f"✓ 数据集已存在: {save_path}")
        return save_path
    
    print(f"📥 下载Robomimic数据集...")
    print(f"  任务: {task} ({dataset_type})")
    print(f"  URL: {url}")
    print(f"  保存到: {save_path}")
    print(f"\n请在终端运行:")
    print(f"  wget {url} -O {save_path}")
    print(f"\n或使用:")
    print(f"  curl -o {save_path} {url}")
    
    return save_path


if __name__ == '__main__':
    # 测试数据加载器
    print("=" * 60)
    print("Robomimic数据加载器测试")
    print("=" * 60)
    
    # 示例：下载数据集
    dataset_path = download_robomimic_dataset(task='square', dataset_type='ph')
    
    # 如果数据集存在，测试加载
    if os.path.exists(dataset_path):
        print("\n测试数据加载...")
        train_loader, val_loader, stats = create_robomimic_dataloaders(
            dataset_path,
            max_episodes=10  # 测试时只加载10条
        )
        
        # 测试一个batch
        for batch in train_loader:
            images, qpos, actions, is_pad = batch
            print(f"\n✓ Batch样本:")
            print(f"  images: {images.shape}")
            print(f"  qpos: {qpos.shape}")
            print(f"  actions: {actions.shape}")
            print(f"  is_pad: {is_pad.shape}")
            break
        
        print("\n✓ 数据加载器测试通过！")
    else:
        print(f"\n⚠️  数据集不存在，请先下载: {dataset_path}")
