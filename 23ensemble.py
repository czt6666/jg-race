"""
23ensemble.py — 候选重排序集成 + 100-way MRR 调参 + 生成提交

流程：
1. 读取各打分器 val/test 分数（itemcf/pop/recent_pop/repeat/sasrec...）。
2. 行内 rank 归一化（每行 100 候选，average rank → [0,1]）。
3. 坐标上升搜索权重，最大化 val 100-way MRR（期望排名平局处理）。
4. 应用到 test，输出 [0,1] 8 位小数提交（加微小确定性 jitter 破平局）。

用法：
  python 23ensemble.py --dataset dataset1 --methods itemcf,sasrec,recent_pop,pop,repeat
"""
import os, os.path as osp, argparse
import numpy as np
from scipy.stats import rankdata


def log(m):
    print(m, flush=True)


def row_rank_norm(scores):
    """[N,100] → 行内 average-rank 归一化到 [0,1]，越大越好。"""
    C = scores.shape[1]
    try:
        r = rankdata(scores, method="average", axis=1)  # scipy>=1.10
    except TypeError:
        r = np.vstack([rankdata(row, method="average") for row in scores])
    return ((r - 1) / (C - 1)).astype(np.float32)


def mrr_expected(pos, neg):
    """期望排名平局处理：rank = 1 + #(neg>pos) + 0.5*#(neg==pos)。"""
    ranks = 1 + np.sum(neg > pos[:, None], axis=1) + 0.5 * np.sum(neg == pos[:, None], axis=1)
    return float(np.mean(1.0 / ranks)), ranks


def blend_mrr(norm_val, weights, col0_is_pos=True):
    """norm_val: dict method->[N,100] (已 rank-norm)。返回 MRR。"""
    keys = list(norm_val.keys())
    S = sum(weights[k] * norm_val[k] for k in keys)
    pos = S[:, 0]; neg = S[:, 1:]
    mrr, _ = mrr_expected(pos, neg)
    return mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scorers_dir", default="./data/scorers")
    ap.add_argument("--methods", default="itemcf,sasrec,recent_pop,pop,repeat")
    ap.add_argument("--sasrec_test_csv", default=None,
                    help="若提供，则用此 csv 作为 test 的 sasrec 分数（候选顺序一致）")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    d = osp.join(args.scorers_dir, args.dataset)
    methods = args.methods.split(",")
    log(f"==== ensemble {args.dataset} methods={methods} ====")

    # ---- 载入 val 分数 ----
    val_raw = {}
    vs = np.load(f"{d}/val_scores.npz")
    for m in methods:
        if m in vs.files:
            val_raw[m] = vs[m]
        elif osp.exists(f"{d}/val_{m}.npz"):
            val_raw[m] = np.load(f"{d}/val_{m}.npz")["score"]
        else:
            log(f"  [warn] val method {m} not found, skip")
    methods = list(val_raw.keys())

    # ---- 行内 rank 归一化（val）----
    log("rank-norm val...")
    norm_val = {m: row_rank_norm(val_raw[m]) for m in methods}

    # 单方法 MRR
    log("--- single-method val MRR ---")
    for m in methods:
        mrr = blend_mrr({m: norm_val[m]}, {m: 1.0})
        log(f"  {m:12s} {mrr:.4f}")

    # ---- 坐标上升搜索权重 ----
    log("--- coordinate ascent on weights ---")
    w = {m: (1.0 if m in ("itemcf", "sasrec") else 0.0) for m in methods}
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    best_mrr = blend_mrr(norm_val, w)
    for _ in range(4):
        for m in methods:
            best_wm, bm = w[m], best_mrr
            for cand in grid:
                w2 = dict(w); w2[m] = cand
                mrr = blend_mrr(norm_val, w2)
                if mrr > bm:
                    bm = mrr; best_wm = cand
            w[m] = best_wm; best_mrr = bm
    log(f"  best weights: { {k: round(v,2) for k,v in w.items()} }")
    log(f"  >>> val 100-way MRR (ensemble) = {best_mrr:.4f}")

    # ---- 应用到 test ----
    log("loading test scores...")
    test_raw = {}
    ts = np.load(f"{d}/test_scores.npz")
    for m in methods:
        if m in ts.files:
            test_raw[m] = ts[m]
        elif osp.exists(f"{d}/test_{m}.npz"):
            test_raw[m] = np.load(f"{d}/test_{m}.npz")["score"]
        elif m == "sasrec" and args.sasrec_test_csv:
            arr = np.loadtxt(args.sasrec_test_csv, delimiter=",", dtype=np.float32)
            test_raw[m] = arr
            log(f"  sasrec test from csv {args.sasrec_test_csv} shape={arr.shape}")
        else:
            log(f"  [warn] test method {m} missing → weight set 0")
            w[m] = 0.0

    use = [m for m in methods if m in test_raw]
    log(f"test methods used: {use}")
    log("rank-norm test...")
    norm_test = {m: row_rank_norm(test_raw[m]) for m in use}
    S = sum(w[m] * norm_test[m] for m in use)

    # min-max 到 [0,1] 行内，加微小确定性 jitter 破平局
    N, C = S.shape
    rmin = S.min(axis=1, keepdims=True); rmax = S.max(axis=1, keepdims=True)
    Snorm = (S - rmin) / np.where(rmax > rmin, rmax - rmin, 1.0)
    # 确定性 hash jitter（基于候选列索引），幅度极小不影响排序主体
    jit = (np.arange(C, dtype=np.float32) % 97) / 97.0 * 1e-6
    Snorm = np.clip(Snorm + jit[None, :], 0.0, 1.0)

    out_csv = args.out_csv or f"{d}/{args.dataset}_result_ensemble.csv"
    with open(out_csv, "w") as f:
        for row in Snorm:
            f.write(",".join(f"{x:.8f}" for x in row) + "\n")
    log(f"saved submission: {out_csv}  shape={Snorm.shape} range=[{Snorm.min():.4f},{Snorm.max():.4f}]")


if __name__ == "__main__":
    main()
