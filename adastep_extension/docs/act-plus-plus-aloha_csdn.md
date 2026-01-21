# 复现笔记：ACT-Plus-Plus (Mobile ALOHA) — 摘自 CSDN（nenchoumi3119）

原文：https://blog.csdn.net/nenchoumi3119/article/details/148724858  （发布日期：2025-06-29，许可：CC BY‑SA）

简要说明
- 本文为作者对 `act-plus-plus` / Mobile ALOHA 的复现教程，含：环境配置、数据格式（HDF5）说明、数据转换（rosbag→hdf5）、仿真数据生成、训练与评估命令、以及作者公开的数据/ckpt 网盘链接。

要点速查
- Conda 环境：`conda create -n aloha python=3.8.10`（或 `python=3.10` 取决 GPU）
- 关键依赖（示例）：`mujoco==2.3.7`, `dm_control==1.0.14`, `robomimic`（特定分支）、`wandb`, `diffusers`, `h5py`, `torch/torchvision` 等。
- HDF5 常见结构：
  - `action` → (T, 14)
  - `observations/images/top` → (T, H, W, 3)
  - `observations/qpos`, `observations/qvel`

可直接运行的命令（摘自文章，已整理）
- 创建环境并安装依赖：

```bash
conda create -n aloha python=3.8.10
conda activate aloha
pip install -r scripts/requirements_aloha.txt
```

- 生成仿真数据（带虚拟屏幕）：

```bash
xvfb-run -s "-screen 0 1400x900x24" python3 record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir "../datas/gen_data" \
  --num_episodes 5
```

- 训练示例：

```bash
python3 imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir "../ckp" \
  --policy_class ACT \
  --kl_weight 1 \
  --chunk_size 10 \
  --hidden_dim 512 \
  --batch_size 1 \
  --dim_feedforward 3200 \
  --lr 1e-5 \
  --seed 0 \
  --num_steps 2000
```

仓库内附件
- `scripts/requirements_aloha.txt` — 从文章提取的依赖清单（可作为独立 env 的安装源）
- `scripts/setup_aloha_env.sh` — 一键创建 conda 环境并安装依赖（含 GPU 注释）
- `scripts/convert_rosbag_to_hdf5.py` — rosbag → hdf5 转换脚本（轻度整理以便重复使用）
- `docs/REPRODUCE_ALOHA.md` — 逐步复现说明（见同目录）

注意事项 & 已知坑
- `robomimic` 的某些分支（如 `diffusion-policy-mg`）在 upstream 可能已移除 — 文章给出了镜像地址与替代策略；复现时若遇到 branch not found，请使用镜像或指定 commit。
- 不同 GPU（50 系）可能需要不同的 Python / PyTorch 轮子（文章给出 nightly 安装示例）。
- 真机数据与仿真数据字段不完全一致（相机视角、关节维度），需要做 `hdf5` 字段映射。 

参考：文章中含大量可运行示例（h5py 浏览、rosbag→hdf5、record/visualize/train/eval）。
