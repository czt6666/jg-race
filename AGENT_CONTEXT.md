# 任务上下文文档（供接手 Agent 阅读）

## 项目概览

这是一个时序图链接预测竞赛。目标是在两个数据集上最大化 MRR 分数：

- **Dataset1 目标**：AP ≥ 0.99（基本已达到）
- **Dataset2 目标**：MRR 尽量高（当前我方 ~0.17，竞争者最高 ~0.43）
- **最终得分** = MRR_dataset1 + MRR_dataset2（满分 2.0，当前约 1.16，目标 1.42+）

---

## 环境信息

- **Python 路径**：`/opt/miniconda/envs/rtdetr_env/bin/python`
- **深度学习框架**：Jittor（不是 PyTorch）
- **工作目录**：`/root/workspace/python/jg-race/`
- **GPU**：两张，CUDA_VISIBLE_DEVICES=0 指定 GPU0（GPU1 有残留显存问题，尽量用 GPU0）
- **GPU 显存**：24576 MiB，当前使用约 10-11 GiB

---

## 数据文件

```
data/
  dataset1/
    train.csv       # 训练边（含 split 列：0=train, 1=val）
    val.csv         # 验证集（有标签）
    test.csv        # 测试集（无标签，提交预测分数）
  dataset2/
    train.csv       # 同上
    val.csv
    test.csv
```

每个 CSV 格式：`src, dst, time, [split]`（split 列只在 train.csv 中）

---

## 核心问题（Dataset2 分数低的根因）

### 1. 冷启动问题（最严重）
- Dataset2 test.csv 的候选 dst 中，**约 54% 是训练集中从未出现过的新 item**
- DyGFormer 对这些 item 的 embedding 是零向量 → 所有新 item 得分相同（约 0.065）
- 这些新 item 的平均排名是 **64.5/100**，MRR ≈ 0.015
- 这是主要的分数天花板

### 2. 验证指标之前严重误导（已修复）
- 原代码 val 用 **pairwise MRR（1 个负样本）**，报告 MRR=0.97+
- 比赛 test 用 **100-way MRR（99 个负样本）**，实际难度天壤之别
- **已修复**：val 现在用 99 个负样本，结果会真实反映 test 难度

### 3. 损失函数（已从 BCE 改为 InfoNCE）
- 原始代码（1DyGFormer.py）用 BCE：正 vs 1负，独立分类，不感知排名
- 当前代码（7DyGFormer.py）用 InfoNCE：正 vs K负，softmax，直接优化排名

---

## 当前运行的实验

```bash
# PID: 1366129（已验证运行中）
CUDA_VISIBLE_DEVICES=0 python -u 7DyGFormer.py \
  --dataset dataset2 \
  --batch_size 200 \
  --max_seq_len 32 \
  --num_neg 3 \
  --p_pop 0.7 \
  --early_stop 3

# 日志：logs/0604/7dyg_dataset2.log
# 速度：~1.2s/it，9976 batch/epoch，约 3.3 小时/epoch
# 状态：Epoch 1 进行中（约 1% 完成，刚重启）
```

---

## 主要代码文件

### `7DyGFormer.py`（当前主文件）
关键设计：
- **InfoNCE loss**：`infonce_loss(pos_logit, neg_logit, K)` 用 cross_entropy 实现
- **PopNegSampler**：70% 热门负样本（sqrt(freq) 加权）+ 30% 随机，更接近 test 分布
- **100-way val MRR**（刚修改）：`test_val()` 用 99 个负样本，分块 forward（每次 10 个）避免 OOM
- **use_all_train**：可选，用 train.csv 全部数据训练（split=0+1）
- **pop_feat**：节点 raw feature 的第一维加入 log(degree+1) 特征

核心参数：
```
--num_neg K      # InfoNCE 负样本数（训练），默认 3
--p_pop 0.7      # PopNegSampler 热门比例
--batch_size 200
--max_seq_len 32
--early_stop 3   # patience（基于 100-way val MRR）
--use_all_train  # 开启后用全部 train.csv 数据（无 val，慎用）
```

### `1DyGFormer.py`（基线，不要修改）
- BCE loss，pairwise 负样本
- Dataset1 Val: MRR=0.9910, AP=0.9926 ✓
- Dataset2 Val: MRR=0.9730, AP=0.9349

### `ANALYSIS.md`
冷启动问题的详细分析，包括数据统计

---

## 已知的坑

1. **GPU1 残留显存**：GPU1 有约 13 GiB 幽灵显存（来自被 kill 的 Jittor 进程），用 GPU0 更稳
2. **K 太大会 OOM**：K=19 时 B×K=3800 对一次 forward → OOM。K=3 时 800 对，安全
3. **GPU 利用率低（~20%）**：PopNegSampler 用 Python 循环采样，是 CPU 瓶颈，未解决
4. **JT_SAVE_MEM=1**：遇到 Jittor segfault 时加此环境变量可能有帮助
5. **neg_sampling_ratio=0**：train_loader 设为 0（不从 DataLoader 采负样本，由 PopNegSampler 手动采）

---

## 优先改进方向（按重要性排序）

### P1：解决冷启动（最大收益）
- 54% 新 item 得分相同，rank=64.5，是最大瓶颈
- 思路：
  - 给新 item 加入 side features（如 item 类目、文本等）——需要看数据是否有
  - 基于热度的 fallback：新 item 按全局热度排序
  - 在 submission 后处理：新 item 用热度/规则重排

### P2：GPU 利用率（加速训练）
- 当前：PopNegSampler 用 Python for 循环 → CPU 卡脖子，GPU 20% 利用率
- 修复：向量化负样本采样，替换 for 循环：
  ```python
  # 当前（慢）：
  for i, (s, d) in enumerate(zip(src, dst)):
      ...
  
  # 目标（快）：
  candidates = np.random.choice(all_items, size=B*K*5, p=pop_probs)
  # 向量化过滤 src==candidates 的情况
  ```
- 预期：GPU 从 20% → 70%+，训练速度提升 3x

### P3：增大 K（更难的排名训练）
- 当前 K=3（4-way ranking），test 是 100-way
- 提升 K 后 InfoNCE 训练更接近 test 难度
- 受限于显存和速度（K 增大，负样本 forward 也增大）
- 如果向量化了 PopNegSampler，可以试 K=7 或 K=15

### P4：增大 batch_size
- 当前 10-11 GiB / 24 GiB，还有约 13 GiB 空余
- 可以尝试 batch_size=400 或 600（同时调整 neg_chunk）

---

## 提交流程

```bash
# 推断脚本在某处，看 saved_models/ 目录
ls saved_models/run_0604/7dyg_dataset2/
# 生成 test 预测后提交（具体提交命令未知，需确认）
```

---

## 当前等待事项

- Epoch 1 完成后查看 **100-way val MRR**（这才是真实指标，预期比 0.9787 低很多，可能 0.3-0.5 左右）
- 根据这个数字决定是否值得继续 or 切换策略
