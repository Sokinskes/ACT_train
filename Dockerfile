# 使用 PyTorch 官方镜像作为基础镜像（支持 CUDA 12.1）
FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn8-devel

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# 更新系统包
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libglvnd0 \
    libx11-dev \
    libxkbcommon0 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    xvfb \
    xauth \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxkbcommon-x11-0 \
    libsdl2-dev \
    libosmesa6-dev \
    libgl1-mesa-dev \
    patchelf \
    libopenmpi-dev \
    openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制项目代码
COPY . /workspace/

# 安装 pip 和必要的 Python 包管理工具
RUN pip install --upgrade pip setuptools wheel

# 使用 Conda 工作流（如果有 environment.yml）
RUN if [ -f environment.yml ]; then \
    pip install conda-pack && \
    conda env create -f environment.yml && \
    conda clean -afy; \
    else \
    pip install -r requirements.txt; \
    fi

# 激活 conda 环境
SHELL ["conda", "run", "-n", "act", "/bin/bash", "-c"]

# 验证关键依赖
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import dm_control; print('dm_control imported successfully')"
RUN python -c "import mujoco; print(f'MuJoCo {mujoco.__version__}')"

# 设置入口点为激活 conda 环境并运行 bash
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "act", "/bin/bash"]

# 提供默认命令
CMD ["-i"]
