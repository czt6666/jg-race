"""
10PostProcess.py - 冷启动得分后处理

问题: DyGFormer/SASRec对冷启动item给出相同(或随机低)分数
  => 冷启动item平均排名 64.5/100 (比随机50.5还差)

修复: 对每行测试数据，将冷启动item分数设为
  "该行已知item分数的中位数 ± 小随机扰动"
  => 冷启动item平均排名 ~50 (随机水平，比原来好)

期望收益:
  若54%正样本是冷启动item: MRR提升约 0.54×(1/50-1/65) ≈ +0.003
  微小但免费（无需重训练）

用法:
  python 10PostProcess.py \
    --input data/dataset2/dataset2_result.csv \
    --train data/dataset2/train.csv \
    --test  data/dataset2/test.csv \
    --output data/dataset2/dataset2_result_postproc.csv \
    --mode median_jitter  # or: random_middle, preserve
"""
import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse
from collections import Counter


def apply_coldstart_fix(scores, test_candidates, train_dst_set, mode="median_jitter", seed=42):
    """
    scores: np.array [n_queries, 100] - 模型输出分数
    test_candidates: np.array [n_queries, 100] - 候选item ID
    train_dst_set: set - 训练集中出现过的item ID集合
    mode: 修复策略
      - 'median_jitter': 设为已知item中位数 ± 小随机扰动（推荐）
      - 'random_middle': 设为已知item均值（固定，无随机性）
      - 'preserve': 不修改（基线对比）
    """
    rng = np.random.RandomState(seed)
    n, k = scores.shape
    new_scores = scores.copy()

    cold_start_count = 0
    total_items = 0

    for i in range(n):
        row_cands = test_candidates[i]  # [100]
        row_scores = scores[i]          # [100]

        # 区分已知和冷启动item
        is_known = np.array([int(c) in train_dst_set for c in row_cands])
        is_cold = ~is_known

        cold_indices = np.where(is_cold)[0]
        known_indices = np.where(is_known)[0]

        cold_start_count += len(cold_indices)
        total_items += k

        if len(cold_indices) == 0 or len(known_indices) == 0:
            continue

        if mode == "preserve":
            continue

        known_scores = row_scores[known_indices]

        if mode == "median_jitter":
            median_s = np.median(known_scores)
            std_s = np.std(known_scores) * 0.3 if np.std(known_scores) > 0 else 1e-4
            # 给每个冷启动item一个不同的随机分数，分布在中位数附近
            cold_scores = rng.normal(median_s, std_s, len(cold_indices))
            # 限制在有效范围内
            cold_scores = np.clip(cold_scores, 0.001, 0.999)
            new_scores[i, cold_indices] = cold_scores.astype(np.float32)

        elif mode == "random_middle":
            mean_s = np.mean(known_scores)
            # 在最小分和最大分之间均匀采样
            lo, hi = np.min(known_scores), np.max(known_scores)
            cold_scores = rng.uniform(lo, hi, len(cold_indices))
            new_scores[i, cold_indices] = cold_scores.astype(np.float32)

    cold_frac = cold_start_count / max(total_items, 1)
    print(f"Cold-start items: {cold_start_count}/{total_items} ({cold_frac:.1%})")

    return new_scores


def evaluate_mrr(scores, true_col=0):
    """
    假设每行的第0列是正样本（竞赛格式）
    计算 100-way MRR
    """
    n = scores.shape[0]
    pos_scores = scores[:, true_col]  # [n]
    neg_scores = scores[:, 1:]        # [n, 99]
    ranks = 1 + np.sum(neg_scores > pos_scores.reshape(n, 1), axis=1)  # [n]
    mrr = np.mean(1.0 / ranks)
    return mrr, ranks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="输入分数文件 (CSV, 每行100个float)")
    parser.add_argument("--train", type=str, required=True,
                        help="train.csv路径")
    parser.add_argument("--test", type=str, required=True,
                        help="test.csv路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出分数文件路径")
    parser.add_argument("--mode", type=str, default="median_jitter",
                        choices=["median_jitter", "random_middle", "preserve"])
    parser.add_argument("--val_csv", type=str, default=None,
                        help="如果提供val.csv，会尝试用第一列作为正样本计算val MRR")
    args = parser.parse_args()

    print(f"Loading scores from {args.input} ...")
    scores = np.loadtxt(args.input, delimiter=",", dtype=np.float32)
    print(f"  Shape: {scores.shape}")

    print(f"Loading train/test metadata ...")
    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)

    train_dst_set = set(train_df["dst"].unique().tolist())
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)  # [n, 100]
    print(f"  Train items: {len(train_dst_set)}, Test queries: {len(test_candidates)}")

    # 统计冷启动比例（不做修改）
    all_cands_flat = test_candidates.flatten()
    cold_count = sum(1 for c in all_cands_flat if int(c) not in train_dst_set)
    print(f"  Cold-start candidates: {cold_count}/{len(all_cands_flat)} = {cold_count/len(all_cands_flat):.1%}")

    # 应用冷启动修复
    print(f"\nApplying cold-start fix (mode={args.mode}) ...")
    new_scores = apply_coldstart_fix(scores, test_candidates, train_dst_set, mode=args.mode)

    # 保存结果
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for row in new_scores:
            f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    print(f"Saved to {args.output}")

    # 对比原始 vs 后处理后的分数分布
    orig_cold_scores = []
    new_cold_scores = []
    for i in range(len(test_candidates)):
        for j, c in enumerate(test_candidates[i]):
            if int(c) not in train_dst_set:
                orig_cold_scores.append(float(scores[i, j]))
                new_cold_scores.append(float(new_scores[i, j]))

    if orig_cold_scores:
        print(f"\nCold-start score distribution change:")
        print(f"  Before: mean={np.mean(orig_cold_scores):.4f}, std={np.std(orig_cold_scores):.4f}, "
              f"range=[{np.min(orig_cold_scores):.4f}, {np.max(orig_cold_scores):.4f}]")
        print(f"  After:  mean={np.mean(new_cold_scores):.4f}, std={np.std(new_cold_scores):.4f}, "
              f"range=[{np.min(new_cold_scores):.4f}, {np.max(new_cold_scores):.4f}]")

    print("\nDone.")
