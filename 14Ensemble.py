"""
14Ensemble.py - 多模型预测集成

对多个 result.csv（每行100个分数）做集成。
排序指标(MRR)下，rank-average 通常比 score-average 更稳健：
  把每行的分数转成排名，再平均排名，最后用 -rank 作为新分数。
也支持 score-average（直接平均 sigmoid 分数）。

用法:
  python 14Ensemble.py --inputs f1.csv f2.csv f3.csv \
    --output ensemble.csv --mode rank   # or score
"""
import numpy as np
import argparse
import os.path as osp


def rank_average(score_list):
    """每个文件每行转排名(降序,最高分=rank1)，平均后取负作为新分数"""
    n, k = score_list[0].shape
    avg_rank = np.zeros((n, k), dtype=np.float64)
    for s in score_list:
        # argsort 降序 → 每个位置的排名
        order = np.argsort(-s, axis=1)          # [n,k] 索引
        rank = np.empty_like(order)
        rows = np.arange(n)[:, None]
        rank[rows, order] = np.arange(1, k + 1)[None, :]
        avg_rank += rank
    avg_rank /= len(score_list)
    # 排名越小越好 → 用 (k - rank) 作为分数，越大越好
    return (k - avg_rank).astype(np.float32)


def score_average(score_list):
    return np.mean(score_list, axis=0).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="各文件权重(仅score模式),默认等权")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["rank", "score"], default="rank")
    args = parser.parse_args()

    print(f"Loading {len(args.inputs)} files...")
    score_list = []
    for f in args.inputs:
        s = np.loadtxt(f, delimiter=",", dtype=np.float32)
        print(f"  {osp.basename(f)}: {s.shape}, range=[{s.min():.4f},{s.max():.4f}]")
        score_list.append(s)

    if args.mode == "rank":
        out = rank_average(score_list)
    else:
        if args.weights:
            w = np.array(args.weights) / np.sum(args.weights)
            out = sum(wi * s for wi, s in zip(w, score_list)).astype(np.float32)
        else:
            out = score_average(score_list)

    print(f"Ensemble ({args.mode}): {out.shape}, range=[{out.min():.4f},{out.max():.4f}]")
    with open(args.output, "w") as fo:
        for row in out:
            fo.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    print(f"Saved: {args.output}")
