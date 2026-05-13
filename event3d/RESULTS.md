# E2V 训练结果

## 环境
- GPU: NVIDIA RTX 3090 (24GB)
- PyTorch: 2.1.0+cu121
- 数据集: SynthEVox3D-Tiny (1040 samples, 13 classes)
- 模型: E2VModel (149M 参数)

## 训练配置
- Epochs: 100 | Batch Size: 5 | LR: 1e-4 | Optimizer: AdamW | Loss: BCEWithLogitsLoss

## Test 集结果 (最优阈值 t=0.10)

| 指标 | 值 |
|------|-----|
| mIoU | **0.3778** |
| F-Score | **0.5343** |

### 逐类 IoU

| 类别 | IoU | 类别 | IoU |
|------|-----|------|-----|
| Airplane | 0.465 | Bench | 0.314 |
| Cabinet | 0.445 | Car | 0.490 |
| Chair | 0.261 | Displayer | 0.409 |
| Lamp | 0.209 | Speaker | 0.269 |
| Rifle | 0.482 | Sofa | 0.404 |
| Table | 0.343 | Telephone | 0.449 |
| Watercraft | 0.372 | | |

## 与论文对比
- 论文 E2V baseline: mIoU 0.346
- 本复现: **0.378 (+3.2%)**

## 优化记录
- 向量化 event_frame_pos/event_frame/build_sae: 数据加载 10-15x 提速
- GPU 利用率: 0% → 100%
- 总训练时间: ~3.5 小时
