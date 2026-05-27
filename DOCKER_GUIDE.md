# Docker 快速入门指南

## 系统要求

### Windows 用户
- **Docker Desktop for Windows** （已安装且正在运行）
- **WSL 2 后端** （推荐）或 **Hyper-V** 
- **最小 8GB RAM**（推荐 16GB+）
- **GPU 支持可选**：需要 NVIDIA GPU + NVIDIA Container Toolkit

### Linux / Mac 用户
- **Docker** 最新版本
- **Docker Compose** 最新版本
- **GPU 支持**（可选）：NVIDIA Docker 运行时

---

## 快速开始

### 方式 1：使用 Docker Compose（推荐）

#### Windows 用户步骤：

1. **安装 Docker Desktop for Windows**
   - 访问 [Docker Desktop 官网](https://www.docker.com/products/docker-desktop)
   - 下载并安装 Windows 版本
   - 确保启用 WSL 2 后端

2. **在项目根目录打开 PowerShell 或 CMD**
   ```powershell
   cd C:\path\to\ACT
   ```

3. **构建并启动容器**
   ```powershell
   docker-compose up -d --build
   ```

4. **进入交互环境**
   ```powershell
   docker-compose exec act-dev conda run -n act bash
   ```

5. **运行评估脚本**
   ```bash
   cd /workspace/third_party/act-plus-plus
   python eval_adastep.py
   ```

#### Linux / Mac 用户步骤：

```bash
cd /path/to/ACT

# 构建镜像
docker-compose up -d --build

# 进入容器
docker-compose exec act-dev conda run -n act bash

# 在容器内运行代码
cd /workspace/third_party/act-plus-plus
python eval_adastep.py
```

---

### 方式 2：直接使用 Docker 命令

#### 1. 构建镜像
```bash
# 开发版（完整，包含编译工具）
docker build -t act:dev -f Dockerfile .

# 运行时版（轻量化）
docker build -t act:runtime -f Dockerfile.runtime .
```

#### 2. 运行容器（交互模式）

**Windows（PowerShell）：**
```powershell
docker run -it --rm `
  -v ${PWD}:/workspace `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/checkpoints:/workspace/checkpoints `
  act:dev
```

**Linux / Mac：**
```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  act:dev
```

#### 3. 运行特定脚本

```bash
docker run --rm \
  -v $(pwd):/workspace \
  act:dev \
  conda run -n act python /workspace/third_party/act-plus-plus/eval_adastep.py
```

---

## GPU 支持配置

### 对于 NVIDIA GPU 用户

#### Windows（Docker Desktop）：
1. 安装 **NVIDIA Container Toolkit**
2. 在 docker-compose.yml 中取消注释 `runtime: nvidia` 行
3. 运行：
   ```powershell
   docker-compose up -d --build
   docker-compose exec act-dev conda run -n act python -c "import torch; print(torch.cuda.is_available())"
   ```

#### Linux：
```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 测试 GPU
docker run --rm --gpus all pytorch/pytorch:latest python -c "import torch; print(torch.cuda.is_available())"
```

---

## 容器内常用命令

```bash
# 进入 act conda 环境
conda activate act

# 查看可用 GPU
nvidia-smi

# 运行评估
cd /workspace/third_party/act-plus-plus
python eval_adastep.py

# 运行训练
python train_adastep.py

# 运行测试
pytest /workspace/tests/

# 退出容器
exit
```

---

## 数据和检查点管理

### 在 Windows 主机上管理数据

```powershell
# 将数据从主机复制到容器
docker cp "C:\local\data\*.hdf5" act-dev:/workspace/data/

# 将输出文件从容器复制回主机
docker cp act-dev:/workspace/checkpoints ./checkpoints_backup
```

### Linux / Mac

```bash
# 复制数据
docker cp ./data/local_file.hdf5 act-dev:/workspace/data/

# 复制输出
docker cp act-dev:/workspace/checkpoints ./checkpoints_backup
```

---

## 常见问题排查

### 问题 1：容器无法启动
**症状**：Docker Compose 报错
**解决方案**：
```bash
# 查看详细日志
docker-compose logs act-dev

# 清理并重建
docker-compose down
docker system prune -a
docker-compose up --build
```

### 问题 2：GPU 未被识别
**症状**：`torch.cuda.is_available()` 返回 False
**解决方案**：
- 确保 NVIDIA Driver 已安装：`nvidia-smi`
- 确保 NVIDIA Container Toolkit 已安装
- 取消注释 docker-compose.yml 中的 `runtime: nvidia`
- 重启 Docker：`docker restart`

### 问题 3：内存不足
**症状**：容器或脚本被杀死（OOM）
**解决方案**：
- 增加 Docker Desktop 内存配置（Settings > Resources）
- 在容器运行时添加内存限制：
  ```bash
  docker run -m 16g ...
  ```

### 问题 4：权限问题（Linux）
**症状**：无法读写挂载的卷
**解决方案**：
```bash
# 以当前用户 ID 运行容器
docker run -it --user $(id -u):$(id -g) ...

# 或修改卷权限
sudo chown -R $(id -u):$(id -g) ./data ./checkpoints
```

---

## Windows 特殊配置

### WSL 2 内存优化

在 `%UserProfile%\.wslconfig` 中添加：
```ini
[wsl2]
memory=16GB
processors=8
swap=4GB
```

### 将 Windows 主机路径挂载到 WSL 2
```powershell
# 在 PowerShell 中
docker run -it -v C:\Users\YourUser\ACT:/workspace act:dev
```

---

## 进阶用法

### 构建自定义镜像
修改 Dockerfile 后重建：
```bash
docker build --no-cache -t act:latest -f Dockerfile .
```

### 使用 Docker Hub（可选）
```bash
# 标记镜像
docker tag act:dev your-docker-hub-id/act:dev

# 推送到 Docker Hub
docker push your-docker-hub-id/act:dev

# 其他用户拉取
docker pull your-docker-hub-id/act:dev
```

### 在后台持久运行
```bash
# 启动一个长时间运行的容器
docker run -d --name act-background \
  -v $(pwd):/workspace \
  act:dev \
  conda run -n act python /workspace/train_adastep.py

# 查看日志
docker logs -f act-background

# 停止
docker stop act-background
```

---

## 性能提示

1. **使用 `.dockerignore`** 减小镜像体积
2. **在容器内使用本地 conda 缓存** 加速后续构建
3. **使用 `--no-cache` 仅在必要时重建**
4. **在 Linux 上使用 BuildKit** 加速构建：
   ```bash
   DOCKER_BUILDKIT=1 docker build ...
   ```

---

## 支持和反馈

如遇问题，请运行以下诊断命令并提交问题：
```bash
docker version
docker-compose version
nvidia-smi  # 如果有 GPU
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock docker:dind docker ps
```
