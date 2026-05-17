# 计图比赛 — 时序图链接预测优化方案

> 任务：给定历史交互三元组 `(src, dst, time)`，对每个测试样本 `(src, time, c1..c100)` 输出 100 个候选目标的链接概率，按 MRR 评分。
> 当前 baseline：`main.py`（CRAFT）、`1DyGFormer.py`（DyGFormer + 多卡 MPI）。

---

## 一、任务与数据再理解

### 1.1 任务本质
- **不是简单的"重复边检测"**：题目明确指出"现有时序图数据集存在重复边比例较高的问题，模型往往依赖对历史连接的记忆"。本赛题刻意提高了**未来新连接**的预测难度。
- **评估指标为 MRR**：正样本在 100 个候选中的倒数排名。优化 BCE / AP 与优化 MRR **不完全等价**——MRR 只看正样本相对排名，对绝对概率不敏感。
- **目标分布即候选分布**：100 个候选构成了一个**受限的负样本集合**，与训练时的随机负采样分布**不一致**——这是当前 baseline 的一大瓶颈。

### 1.2 数据集统计（实测）

| 指标 | dataset1 | dataset2 |
|---|---|---|
| 二部图 | 否 | 是 |
| 节点 / 边 | 23k / 691k | 63k(12.7k+50.6k) / 2.26M |
| 时间跨度 | 0 ~ 1.30e8（train），1.30e8 ~ 1.35e8（test） | 9.55e8 ~ 1.30e9（train），1.30e9 ~ 1.33e9（test） |
| 边重复率 (count>1) | **52.66%**（mean 3.65, max 705） | **2.26%**（max 2） |
| src↔dst 反向边 | ~64% 对称 | 不存在（二部图） |
| **c1 是 src 历史邻居的比例** | **0.72%** | **1.66%** |
| 100 候选中 src 历史邻居数 | **0.75** | **1.67** |
| Cold-start test src | 95 / 12112 | 0 / 2180 |
| 每个 test src 的查询数 | — | median 21, max 3714 |

### 1.3 关键洞察
1. 正样本几乎都是**新邻居**——纯记忆模型（EdgeBank、纯 popularity）无效。
2. dataset1 高度对称且重复率高 → **co-occurrence、二阶邻居、社区结构** 信号强。
3. dataset2 严格一阶单次交互 → **协同过滤（item-item co-occurrence、user-user 相似）+ 时序新颖性** 是主信号。
4. 当前 `compute_src_dst_node_temporal_embeddings(src, dst, t)` 在 100 候选场景下被调用 100 次/查询 → 推理是 O(100×N)，**有结构性优化空间**。
5. 训练时随机采负 vs. 评测时 100 个"看似合理"的候选 → **分布不匹配**，难负样本挖掘是高 ROI 改进。

---

## 二、优化方法清单（按优先级）

### 🔴 高优先级（预期 MRR 大幅提升，实现成本中等）

#### H1. 难负样本采样（Hard Negative Mining）
**问题**：`TemporalDataLoader` 默认在全集中均匀采负，得到的负样本与"100 个仔细挑选的候选"在分布上有巨大 gap。
**做法**：
- **流行度负采样**：按 `dst` 度数采样，让训练分布贴近测试候选（测试候选大概率从热门项中抽）。
- **二跳邻居负采样**：从 src 的 friend-of-friend 中采负（对 dataset1 尤其有效）。
- **In-batch negatives**：把同一 batch 里其他正样本的 dst 当作 src 的负——免费扩大负样本数。
- **混合采样**：`p_uniform=0.3, p_popularity=0.4, p_2hop=0.3`，可调。
**预期收益**：MRR +0.03~0.08。**实现成本**：~100 行修改 `TemporalDataLoader` 或包装一层。
**风险**：太"难"的负样本会让模型崩溃，需要 warmup（前 N 个 epoch 仍用均匀采样）。

#### H2. 损失函数从 BCE 切换为 ranking loss
**问题**：BCE 优化的是逐对二分类，与 MRR 评分（100 个里排第几）不对齐。
**做法**：
- **多负样本 BPR / Softmax ranking loss**：一次构造 `(src, pos, neg_1, ..., neg_K)`，K=20~99，用 `-log( exp(s_pos) / Σ exp(s_i) )`。
- 或 **InfoNCE / SimCLR-style**：温度 τ 可调（0.05~0.2）。
- 推理时直接用 logits 排序，**省掉 sigmoid**（数值更稳）。
**预期收益**：MRR +0.02~0.05，**且**让模型训练分布更接近测试。
**实现成本**：低，~30 行修改 `train()` 循环。
**与 H1 协同**：H1+H2 合起来上限更高。

#### H3. 引入手工统计特征 + 模型分数融合
**做法**：训练一个轻量 GBDT / LR 把以下信号融进最终分数：
- `is_historical_neighbor(src, c)`：c 是否在 src 历史邻居里
- `last_interact_gap(src, c, t)`：上次交互到 t 的时间差（带指数衰减）
- `interact_count(src, c)`：历史交互次数
- `dst_popularity(c, window=[t-Δ, t])`：c 近期流行度（time-decay）
- `common_neighbors(src, c)`：共同邻居数（2-hop）
- `jaccard(src_neighbors, c_neighbors)`：相似度
- `edge_pmi(src, c)`：归一化的共现 PMI
- DyGFormer 输出的 logit
**融合**：`final = α · σ(logit) + (1-α) · learned_score`，α 用验证集 grid search。
**预期收益**：在 100 候选 reranking 上 MRR +0.02~0.06。
**实现成本**：中等，需要一份离线特征流水线（~200 行 NumPy）。
**理由**：100 候选 reranking 本质就是 LTR 问题，DyGFormer 看不到的图结构特征（共同邻居、PMI）极其互补。

#### H4. 模型集成（DyGFormer + GraphMixer + TGN）
**做法**：JittorGeometric 已经提供 5 个时序模型。训练 3 个不同结构的模型，分数取**rank 平均**（不是概率平均，对 MRR 友好）：
```python
final_rank = (rank_dygformer + rank_graphmixer + rank_tgn) / 3
```
**为什么是 rank 而不是 prob**：不同模型的 logit 尺度不一致，rank 不受影响；MRR 本身基于 rank。
**预期收益**：MRR +0.01~0.04（取决于模型异质性）。
**实现成本**：中等，跑多套训练；推理时三模型输出 + 简单 rank 融合。
**注意**：dataset2 用 TGN 时 memory 模块会撑爆显存，需要分批 fan-in。

#### H5. EdgeBank 兜底分数
**做法**：实现一个 0 参数的 EdgeBank 基线（"过去 Δ 时间窗口内 src 是否与 c 交互过"），作为**特征加入 H3 融合**或**线性兜底**。
**理由**：虽然 c1 是历史邻居的概率只有 0.7% / 1.66%，但**一旦命中，rank 几乎一定靠前**，是高精度低召回的特征——非常适合做 fallback boost。
**预期收益**：MRR +0.005~0.015（小但稳）。
**实现成本**：极低，~50 行。

---

### 🟡 中优先级（局部改进，3~5% 增益范围）

#### M1. 节点 / 边特征改造
当前 baseline 使用 `np.zeros((node_size, dim))` 全零特征。换成：
- **可学习 Embedding**：`nn.Embedding(node_size, dim)`，让节点 ID 进入梯度更新（DyGFormer 内部其实已经这么做了，但维度可调）。
- **度数 / 流行度 buckets**：把节点按出入度分桶（log scale）作为初始特征。
- **时间编码改进**：使用 `Mercer time encoding` 或 `Bochner features`，对超长时间跨度（dataset2 跨度 3.5 亿）更稳。

#### M2. 训练 / 验证集划分策略
当前是按 train.csv 末尾 15% 做验证，但 test.csv 时间紧接 train 结束（dataset1 gap 仅 353）→ **验证集应该是 train 的最后一段**（已经是了，OK），但要进一步：
- **rolling window val**：跑两次，分别用最后 10% 和最后 20% 做 val，取 ensemble。
- **训练时把 train+val 全部喂给 neighbor sampler**（当前已是 `full_data`，OK），但**损失只在前 85% 算**，防止信息泄露。

#### M3. 邻居采样策略
当前 `get_neighbor_sampler(full_data, 'recent', seed=1)`。可以试：
- `'time_interval_aware'`：按时间间隔加权
- `'uniform'`：均匀采样作为 ensemble 中的一支
- 增大 `max_seq_len` 到 64/128（dataset2 src 出度均值 178，30 邻居太少）

#### M4. Test-time augmentation (TTA)
**做法**：对同一 query 用不同的 neighbor sample seed 跑多次，分数取平均。
**预期**：MRR +0.005~0.015，但推理时间 ×3~5。
**值得吗**：评测只看一次分数提交，可以做；但和 H4 集成有重叠。

#### M5. 推理 batching 改造（性能优化，不直接涨 MRR 但能跑更大模型）
当前 `test_competition` 里 `np.repeat(src, 100)` → 把 src 复制 100 份过 backbone。  
**优化**：DyGFormer 的 src embedding 在 100 个候选里是**完全一样的**，应该缓存：  
```python
src_emb = backbone.compute_node_embeddings(src, t)  # 1 次
dst_emb = backbone.compute_node_embeddings(cands.flatten(), t.repeat(100))  # 100 次
logit = predictor(src_emb.repeat(100), dst_emb)
```
**预期**：推理速度 ~2×，省下来的预算可以用在 H4 集成上。  
**前提**：要看 `DyGFormer.compute_src_dst_node_temporal_embeddings` 是否真的对 src 做了无关于 dst 的独立计算（通常是的）；如果有 src-dst cross attention，则缓存失败。

#### M6. 学习率 / 调度器
- 当前 `lr=1e-4` 固定 → 加 **cosine decay + warmup**。
- 对 dataset2 大数据集 → `lr=3e-4` 起步 + linear warmup（5% steps）+ cosine。
- DyGFormer 论文用的也是 1e-4，但配合 AdamW + weight_decay=1e-5 会更稳。

#### M7. 模型规模调优
- `node_feat_dim` 128 → 256（dataset2 节点多）
- `num_layers` 2 → 3
- `num_heads` 2 → 4
- `max_input_sequence_length` 32 → 64
- 注意显存，单卡可能撑不住，配合 MPI 多卡。

---

### 🟢 低优先级（边际收益、长尾改进）

#### L1. 全图二跳特征预计算（GCN-style）
- 跑一次离线 LightGCN / SimpleGCN 得到节点静态 embedding，concat 到 DyGFormer 输入。
- 风险：dataset2 节点数 6 万，可行；但和 DyGFormer 的可学习 embedding 重叠。

#### L2. 对比学习预训练
- 用未来一步预测 / 节点掩码 / 边类型预测做 self-supervised 预训练，再 finetune。
- 论文上 DyGFormer 在 small data 上有收益，但 dataset2 数据本身已经够多。

#### L3. 模型蒸馏 → 一个轻量学生模型
- 比赛只评测一次，没必要为了部署蒸馏；如果集成模型 inference 跑不完才考虑。

#### L4. 异常 / 长尾 src 特殊处理
- dataset2 有的 src 有 3714 个 test 查询，可以为他们用更长的历史窗口；为只有 1-2 个查询的 src 用快速规则。
- 实现复杂，收益不确定，留到后期消耗剩余预算时尝试。

#### L5. 双向图利用（仅 dataset1）
- dataset1 64% 对称 → 把 (src→dst) 和 (dst→src) 都视为同一图，邻居采样时合并。
- DyGFormer 本身可能已经处理；要确认 `bipartite=False` 时是否对称建邻居表。

#### L6. 数据增强：边时间扰动 / dropout
- 训练时随机 drop 10% 历史边或扰动时间戳 ±ε，提升泛化。
- 风险：会破坏时序信号。

#### L7. 输出概率校准
- MRR 只看 rank，不需要校准；但如果做集成融合，温度 scaling 可以让多模型 logits 更可加。
- 用验证集拟合一个 sigmoid 缩放 `σ(a·logit + b)`。

---

## 三、推荐执行顺序

**第一周**（高 ROI 单点改进）
1. M5 推理 batching 重构（节省后续 ablation 的时间）✅ 半天
2. H2 ranking loss + multi-negative ✅ 1 天
3. H1 难负样本采样（先做 popularity-based）✅ 1 天
4. 在 dataset1 上消融验证 H1+H2，目标 MRR > baseline + 0.05

**第二周**（互补特征 + 集成）
5. H3 手工特征流水线 + GBDT 重排 ✅ 2 天
6. H4 多模型集成（DyGFormer + GraphMixer）✅ 2 天
7. H5 EdgeBank 兜底（作为 H3 的特征）✅ 半天

**第三周**（精调）
8. M3/M6/M7 邻居数、学习率、模型规模 sweep ✅ 2 天
9. 在 dataset2 上重新跑全套（部分配置可能要调）✅ 2 天

**关键检查点**：每个改动都要**单独在 dataset1 val MRR 上量化收益**，避免堆功能但没涨分。

---

## 四、风险与注意事项

1. **B 榜数据量级 ×10**（百万节点、千万边）→ 高优 H 系列方案都要**在 dataset2 上验证可扩展性**（H3 共同邻居计算最容易 OOM，需要稀疏化）。
2. **答辩占 35%**：方法的**新意 + 解释清晰度**比 MRR 多挤 0.001 更重要 → H3 的可解释特征 + H4 的集成讲故事比无脑调参好。
3. **Jittor 版本兼容**：`1DyGFormer.py` 里已经 monkey-patch 了 `pad_sequence`，新增方法时小心同类问题。
4. **MPI 多卡保存路径冲突**：当前文件 IO 已经做了 rank-0 守卫，新增 checkpoint 不要忘记。

---

## 五、立刻可以做的最小改动（如果时间紧）

按 ROI 从高到低，**仅做这三件事**就能稳拿 +0.05~0.10 MRR：

1. **H2 → ranking loss**（30 行）
2. **H1 → popularity-based 难负采样**（100 行）
3. **H3 → 加 3 个手工特征：`historical_neighbor`、`common_neighbors`、`popularity_recent`**（150 行）+ 线性融合

合计约 1 个工作日，可作为后续所有优化的基线。
