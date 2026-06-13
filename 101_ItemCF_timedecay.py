"""
101_ItemCF_timedecay.py — Item-based Collaborative Filtering with Time Decay

算法原理：
  - 构建物品-物品共现相似度矩阵（用 scipy sparse）
  - 对每个测试查询，取用户历史最近 N 个物品，计算候选的 ItemCF 分数
  - 时间衰减：老的历史权重低，新的历史权重高
  - 热度去偏：减去 alpha * log(item_count + 1)
  - 无需 GPU 训练，纯 CPU 计算

适合 Dataset2 的理由：
  - 几乎无重复边（2.2%），用户兴趣靠相似物品推荐
  - 测试用户历史很长（中位 1037），协同过滤信号丰富
  - ItemCF 与 SASRec 互补（序列 vs 协同）

输出目录：101_ItemCF_timedecay_MMDD
"""
import os
import os.path as osp
import sys
import time
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--history_len", type=int, default=200,
                    help="用于打分的用户历史长度")
parser.add_argument("--alpha_pop", type=float, default=0.3,
                    help="热度去偏系数：score -= alpha * log(item_count+1)")
parser.add_argument("--decay_tau", type=float, default=30,
                    help="时间衰减半衰期（天）：weight = exp(-delta_days/tau)")
parser.add_argument("--top_k_sim", type=int, default=300,
                    help="每个物品保留最相似的 K 个物品（稀疏截断）")
parser.add_argument("--cold_score", type=float, default=0.0,
                    help="冷启动物品的默认打分偏置（0=不调，正=抬高冷启动排名）")
args = parser.parse_args()

date_str = datetime.now().strftime("%m%d")
script_name = osp.splitext(osp.basename(__file__))[0]
if args.output_dir is None:
    args.output_dir = osp.join(".", f"{script_name}_{date_str}")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(osp.join(args.output_dir, args.dataset), exist_ok=True)

print(f"=== {script_name} ===", flush=True)
print(f"Dataset: {args.dataset}, output: {args.output_dir}", flush=True)

# ============================================================
# Load data
# ============================================================
data_path = osp.join(args.data_dir, args.dataset)
df = pd.read_csv(osp.join(data_path, "train.csv"))
test_df = pd.read_csv(osp.join(data_path, "test.csv"))

src_all = df["src"].values.astype(np.int64)
dst_all = df["dst"].values.astype(np.int64)
t_all = df["time"].values.astype(np.float64)

test_src = test_df["src"].values.astype(np.int64)
test_time = test_df["time"].values.astype(np.float64)
test_cands = test_df.iloc[:, 2:].values.astype(np.int64)  # [N, 100]
N_test, N_cands = test_cands.shape

print(f"Train: {len(df)} edges, Test: {N_test} queries, Cands: {N_cands}", flush=True)

# ============================================================
# 统计训练物品
# ============================================================
all_train_items = np.unique(dst_all)
n_train_items = len(all_train_items)
max_item_id = int(max(dst_all.max(), test_cands.max()))

# 物品 → 训练出现次数（热度）
item_count = np.zeros(max_item_id + 1, dtype=np.int32)
for d in dst_all:
    item_count[d] += 1

print(f"Training items: {n_train_items}, Max item ID: {max_item_id}", flush=True)

# ============================================================
# 构建用户历史（按时间排序）
# ============================================================
print("Building user history...", flush=True)
user_history = defaultdict(list)
for s, d, ts in zip(src_all, dst_all, t_all):
    user_history[int(s)].append((float(ts), int(d)))
for uid in user_history:
    user_history[uid].sort()  # 按时间排序

# ============================================================
# 构建 item-user 稀疏矩阵（用于 ItemCF）
# ============================================================
print("Building item-user matrix...", flush=True)

# 对用户 ID 重映射（0-indexed）
unique_users = np.unique(src_all)
user2idx = {int(u): i for i, u in enumerate(unique_users)}
n_users = len(unique_users)

# 稀疏矩阵 (n_items, n_users)：每个非零表示该用户买过该物品
rows = dst_all.astype(np.int64)
cols = np.array([user2idx[int(s)] for s in src_all], dtype=np.int64)
# 用 1/sqrt(user_degree) 加权（IILE: inverse user length）
user_degree = np.bincount(cols, minlength=n_users).astype(np.float32)
user_weights = 1.0 / np.sqrt(np.maximum(user_degree[cols], 1))
item_degree = np.bincount(rows, minlength=max_item_id + 1).astype(np.float32)
item_weights = 1.0 / np.sqrt(np.maximum(item_degree[rows], 1))
data_weights = user_weights * item_weights

X = csr_matrix(
    (data_weights, (rows, cols)),
    shape=(max_item_id + 1, n_users),
    dtype=np.float32
)
print(f"  Item-user matrix: {X.shape}, nnz={X.nnz:,}", flush=True)

# ============================================================
# 时间窗口热度特征（辅助信号）
# ============================================================
# 计算最大时间（训练集结束时间）
t_max = t_all.max()
DAY = 86400.0
# recent_count[item] = 过去 365 天内出现次数
mask_365 = t_all >= (t_max - 365 * DAY)
recent_count_365 = np.zeros(max_item_id + 1, dtype=np.float32)
for d, m in zip(dst_all, mask_365):
    if m:
        recent_count_365[d] += 1

mask_90 = t_all >= (t_max - 90 * DAY)
recent_count_90 = np.zeros(max_item_id + 1, dtype=np.float32)
for d, m in zip(dst_all, mask_90):
    if m:
        recent_count_90[d] += 1

print(f"Recent 365d items: {(recent_count_365>0).sum():,}, 90d: {(recent_count_90>0).sum():,}", flush=True)

# ============================================================
# 打分函数
# ============================================================
def score_query_batch(batch_src, batch_time, batch_cands):
    """
    为一批查询打分
    batch_src: [B] int64 用户 ID
    batch_time: [B] float64 查询时间
    batch_cands: [B, 100] int64 候选物品 ID
    返回: [B, 100] float32 分数
    """
    B = len(batch_src)
    scores = np.zeros((B, N_cands), dtype=np.float32)

    for i in range(B):
        uid = int(batch_src[i])
        qt = float(batch_time[i])
        cands = batch_cands[i]

        # 获取用户历史（截止 qt 之前）
        hist = user_history.get(uid, [])
        # 过滤时间
        hist_before = [(ts, item) for ts, item in hist if ts < qt]
        if len(hist_before) == 0:
            # 无历史：用全局热度
            for j, c in enumerate(cands):
                if c <= max_item_id:
                    pop = np.log1p(item_count[c]) * 0.5
                    scores[i, j] = pop - args.alpha_pop * np.log1p(item_count[c])
            continue

        # 取最近 history_len 个
        if len(hist_before) > args.history_len:
            hist_before = hist_before[-args.history_len:]

        # 时间衰减权重
        tau_sec = args.decay_tau * DAY
        hist_items = []
        hist_weights = []
        for ts, item in hist_before:
            delta = max(0.0, qt - ts)
            w = np.exp(-delta / tau_sec)
            hist_items.append(item)
            hist_weights.append(w)

        hist_items_arr = np.array(hist_items, dtype=np.int64)
        hist_weights_arr = np.array(hist_weights, dtype=np.float32)
        hist_weights_arr /= (hist_weights_arr.sum() + 1e-8)

        # 用户兴趣向量：历史物品嵌入的加权和
        # user_interest = Σ_h w(h) * X[h]  (稀疏行加权和)
        valid_mask = hist_items_arr <= max_item_id
        valid_items = hist_items_arr[valid_mask]
        valid_weights = hist_weights_arr[valid_mask]

        if len(valid_items) == 0:
            continue

        # 物品向量加权和 → 稀疏格式
        user_vec = X[valid_items].T.dot(valid_weights)  # [n_users,]

        # 对每个候选打分
        for j, c in enumerate(cands):
            c = int(c)
            if c <= max_item_id and item_count[c] > 0:
                # ItemCF 分数：候选向量 · 用户兴趣向量
                cand_vec = X[c]
                icf_score = float(cand_vec.dot(user_vec))

                # 时间热度加成
                pop_365 = np.log1p(recent_count_365[c]) * 0.2
                pop_90 = np.log1p(recent_count_90[c]) * 0.1

                # 热度去偏
                pop_debiase = args.alpha_pop * np.log1p(item_count[c])

                scores[i, j] = icf_score + pop_365 + pop_90 - pop_debiase
            else:
                # 冷启动物品：给一个小的正偏置（防止它们被完全淹没）
                scores[i, j] = args.cold_score

    return scores


# ============================================================
# 生成预测
# ============================================================
print("\nScoring test queries...", flush=True)
all_scores = np.zeros((N_test, N_cands), dtype=np.float32)

BATCH_SIZE = 500
t_start = time.time()
for i in tqdm(range(0, N_test, BATCH_SIZE), ncols=100, desc="Scoring"):
    end = min(i + BATCH_SIZE, N_test)
    batch_scores = score_query_batch(
        test_src[i:end],
        test_time[i:end],
        test_cands[i:end]
    )
    all_scores[i:end] = batch_scores

elapsed = time.time() - t_start
print(f"Scoring done in {elapsed:.1f}s", flush=True)

# ============================================================
# 归一化分数到 [0, 1]（Sigmoid 归一化）
# ============================================================
all_scores_min = all_scores.min(axis=1, keepdims=True)
all_scores_max = all_scores.max(axis=1, keepdims=True)
eps = 1e-8
all_scores_norm = (all_scores - all_scores_min) / (all_scores_max - all_scores_min + eps)

# ============================================================
# 本地验证（val split）
# ============================================================
if "split" in df.columns:
    val_df = df[df["split"] == 1].reset_index(drop=True)
    if len(val_df) > 0:
        print("\nLocal validation (split=1, 99 random neg)...", flush=True)
        rng = np.random.RandomState(42)
        all_items = np.unique(dst_all)
        val_src_arr = val_df["src"].values.astype(np.int64)
        val_dst_arr = val_df["dst"].values.astype(np.int64)
        val_t_arr = val_df["time"].values.astype(np.float64)

        VAL_N = min(5000, len(val_df))
        idx = rng.choice(len(val_df), VAL_N, replace=False)
        val_src_sample = val_src_arr[idx]
        val_dst_sample = val_dst_arr[idx]
        val_t_sample = val_t_arr[idx]

        # 采 99 随机负样本
        neg_matrix = rng.choice(all_items, size=(VAL_N, 99), replace=True).astype(np.int64)
        val_cands = np.concatenate([
            val_dst_sample.reshape(-1, 1),
            neg_matrix
        ], axis=1).astype(np.int64)

        val_scores = score_query_batch(val_src_sample, val_t_sample, val_cands)
        val_scores_min = val_scores.min(axis=1, keepdims=True)
        val_scores_max = val_scores.max(axis=1, keepdims=True)
        val_scores = (val_scores - val_scores_min) / (val_scores_max - val_scores_min + eps)

        # 正样本在第 0 列
        pos_score = val_scores[:, 0]
        neg_scores = val_scores[:, 1:]
        ranks = 1 + np.sum(neg_scores > pos_score.reshape(-1, 1), axis=1)
        mrr = float(np.mean(1.0 / ranks))
        print(f"Local val MRR (100-way, random neg): {mrr:.4f}", flush=True)

# ============================================================
# 保存结果
# ============================================================
out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_itemcf.csv")
with open(out_file, "w") as f:
    for row in all_scores_norm:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
print(f"\nSaved: {out_file}", flush=True)
print(f"Scores shape: {all_scores_norm.shape}, "
      f"range=[{all_scores_norm.min():.4f}, {all_scores_norm.max():.4f}]", flush=True)
print("DONE", flush=True)
