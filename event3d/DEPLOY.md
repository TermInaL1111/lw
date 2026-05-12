# E2V 复现 — AutoDL 部署完整指南

## 概述

在 RTX 3090 (24GB) 上复现 DenseVoxel E2V 论文（Chen et al., ICVR 2023）的完整流程。

## 第一步：上传数据集到 AutoDL 网盘

AutoDL 网盘路径 `/root/autodl-fs/` 是所有实例共享的持久存储。

在**本地 Windows** 上，用 SCP 上传（替换 `<port>` 和 `<host>` 为实际的）：

```bash
# 先在 AutoDL 网盘创建目录
ssh -p <port> root@<host> "mkdir -p /root/autodl-fs/synthevox3d"

# 上传 SynthEVox3D-Tiny（约 2GB）
scp -rP <port> "D:/cugdocuments/科研/datasets/synthevox3d/SynthEVox3D-Tiny" \
    root@<host>:/root/autodl-fs/synthevox3d/
```

**或**用 AutoDL 网页端的文件管理上传（适合网络不稳定时）。

**验证**：登录实例后运行：
```bash
ls /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny/
# 应看到: event_3d_scan_tiny/  data_split/
```

## 第二步：登录 AutoDL 实例并克隆代码

```bash
# 1. SSH 登录（AutoDL 控制台提供命令）
ssh -p <port> root@<host>

# 2. 克隆仓库
cd /root
git clone https://github.com/TermInaL1111/lw.git
# 或 SSH:
# git clone git@github.com:TermInaL1111/lw.git

cd /root/lw
```

## 第三步：安装环境

### 方式 A：一键脚本
```bash
cd /root/lw
bash event3d/setup_autodl.sh
```

### 方式 B：手动安装
```bash
# 创建 conda 环境
conda create -n e2v python=3.10 -y
conda activate e2v

# 安装 PyTorch（根据 AutoDL 实例的 CUDA 版本选择）
# CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
cd /root/lw
pip install numpy scipy matplotlib pyyaml tqdm tensorboard

# 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

## 第四步：启动训练

```bash
cd /root/lw
conda activate e2v

# 一键训练
bash event3d/run_train.sh

# 或手动指定参数
python event3d/train.py \
    --data_root /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny \
    --split_csv /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv \
    --config event3d/configs/default.yaml \
    --device cuda \
    --batch_size 5 \
    --epochs 100

# 后台运行（防止 SSH 断开中断）：
nohup python event3d/train.py \
    --data_root /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny \
    --split_csv /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv \
    --config event3d/configs/default.yaml \
    --device cuda \
    --batch_size 5 \
    --epochs 100 > train.log 2>&1 &

# 查看进度
tail -f train.log
```

## 第五步：评估结果

```bash
# 用最佳 checkpoint 评估测试集
python event3d/evaluate.py \
    --checkpoint event3d/checkpoints/best_model.pth \
    --data_root /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny \
    --split_csv /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv \
    --split test

# 指定特定 epoch checkpoint
python event3d/evaluate.py \
    --checkpoint event3d/checkpoints/e2v_epoch050.pth \
    --data_root /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny \
    --split_csv /root/autodl-fs/synthevox3d/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv \
    --split test
```

## 第六步：查看 TensorBoard

在 AutoDL 实例上启动 TensorBoard：
```bash
tensorboard --logdir event3d/logs --port 6006 --bind_all
```
然后在 AutoDL 控制台找到「自定义服务」→ 添加 6006 端口映射 → 浏览器打开。

## 预期结果

| 指标 | 论文目标 | 备注 |
|------|---------|------|
| mIoU (test) | ~0.346 (full) / ~0.358 (Tiny) | E2V EventFrame(Pos) baseline |
| F-Score@20% | ~0.112 (full) / ~0.507 (Tiny) | |
| 训练时间 | ~4-6h (RTX 3090, 100 epochs) | batch_size=5, SynthEVox3D-Tiny |
| 最佳阈值 | ~0.22 | 自动搜索 |

## GPU 配置建议

| GPU | Batch Size | 预计显存 | Epoch 时长 |
|-----|-----------|---------|-----------|
| RTX 3090 24GB | 5-8 | ~8-12GB | ~3-5 min |
| RTX 4090 24GB | 8-12 | ~10-16GB | ~2-3 min |
| A100 40GB | 16-24 | ~20-35GB | ~1-2 min |
| V100 32GB | 8-12 | ~12-20GB | ~5-8 min |

## 常见问题

**Q: OOM (Out of Memory)？**
A: 减小 batch_size（`--batch_size 2` 或 `--batch_size 1`）

**Q: CUDA 版本不匹配？**
A: 检查 `nvcc --version`，选择对应的 PyTorch 版本：
- CUDA 11.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

**Q: SSH 断开后训练中断？**
A: 使用 `nohup ... &` 或 `tmux`/`screen` 保证后台运行

**Q: 数据路径不对？**
A: 检查 `--data_root` 指向包含 `event_3d_scan_tiny/` 的目录
