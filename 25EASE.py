"""
25EASE.py — EASE (Embarrassingly Shallow AutoEncoder) 协同过滤
            闭式 item-item 岭回归，业界常胜过神经 CF。

模型：B = -P/diag(P), P=(X^T X + λI)^{-1}, 对角置0。
打分：score(user, j) = Σ_{i∈user历史} B[i,j]  (item-item 全局协同，可泛化新边)

评估：诚实 100-way harness（21scorers 生成的 val_sets.npz）。
冷物品(不在所选词表)→分0。

用法：
  python 25EASE.py --dataset dataset2 --topk_items 40000 --lam 250
"""
import os, os.path as osp, argparse, time
import numpy as np
import pandas as pd
import scipy.sparse as sp


def log(m): print(m, flush=True)


def mrr_expected(pos, neg):
    r = 1 + np.sum(neg > pos[:, None], 1) + 0.5 * np.sum(neg == pos[:, None], 1)
    return float(np.mean(1.0 / r))


def build_B(hist_df, item_ids, lam, weighted=False):
    """用 hist_df 构建 EASE 的 B 矩阵（仅 item_ids 子集）。"""
    item2col = {it: c for c, it in enumerate(item_ids)}
    uids = hist_df.src.values
    dids = hist_df.dst.values
    # 过滤到所选物品
    mask = np.isin(dids, item_ids)
    uids = uids[mask]; dids = dids[mask]
    uniq_u, u_idx = np.unique(uids, return_inverse=True)
    d_idx = np.array([item2col[int(d)] for d in dids], dtype=np.int64)
    n_u = len(uniq_u); n_i = len(item_ids)
    data = np.ones(len(u_idx), np.float32)
    X = sp.csr_matrix((data, (u_idx, d_idx)), shape=(n_u, n_i))
    X.data[:] = 1.0  # 二值
    log(f"  X: {X.shape}, nnz={X.nnz}")
    t0 = time.time()
    G = (X.T @ X).toarray().astype(np.float64)  # [n_i, n_i]
    log(f"  G built {G.shape} in {time.time()-t0:.1f}s; inverting...")
    diag_idx = np.diag_indices(n_i)
    G[diag_idx] += lam
    t0 = time.time()
    P = np.linalg.inv(G)
    log(f"  inverted in {time.time()-t0:.1f}s")
    B = P / (-np.diag(P))[None, :]
    B[diag_idx] = 0.0
    return B.astype(np.float32), item2col


def score_cands(B, item2col, hist, src_arr, time_arr, cands):
    """score(c) = Σ_{i∈user历史(<time)} B[i, c]，仅索引候选列(快)。"""
    N, C = cands.shape
    out = np.zeros((N, C), np.float32)
    for r in range(N):
        s = int(src_arr[r]); t = float(time_arr[r])
        seq = hist.get(s, [])
        rows = [item2col[d] for (tt, d) in seq if tt < t and d in item2col]
        if not rows:
            continue
        ccols = np.array([item2col.get(int(c), -1) for c in cands[r]])
        valid = ccols >= 0
        if not valid.any():
            continue
        sub = B[np.ix_(rows, ccols[valid])]  # [h, n_valid]
        out[r, valid] = sub.sum(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--topk_items", type=int, default=40000)
    ap.add_argument("--lam", type=float, default=250.0)
    ap.add_argument("--scorers_dir", default="./data/scorers")
    ap.add_argument("--score_test", action="store_true")
    ap.add_argument("--tag", default="ease")
    args = ap.parse_args()

    sc_dir = osp.join(args.scorers_dir, args.dataset)
    tr = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
    te = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")
    cand_cols = [c for c in te.columns if c.startswith("c")]

    if "split" in tr.columns:
        hist_df = tr[tr.split == 0].reset_index(drop=True)
    else:
        tr = tr.sort_values("time", kind="stable").reset_index(drop=True)
        hist_df = tr.iloc[:-int(len(tr) * 0.15)].reset_index(drop=True)
    full_df = tr

    # 选 top-k 频繁物品（warm）
    freq = hist_df.dst.value_counts()
    item_ids = np.array(freq.index[:args.topk_items], dtype=np.int64)
    log(f"{args.dataset}: {len(item_ids)} items (of {hist_df.dst.nunique()} warm), lam={args.lam}")

    # 用户历史（val 用 hist_df）
    from collections import defaultdict
    def build_hist(df):
        h = defaultdict(list)
        for s, d, t in zip(df.src.values, df.dst.values, df.time.values):
            h[int(s)].append((float(t), int(d)))
        for s in h: h[s].sort()
        return h

    log("building B (val, hist_df)...")
    B, item2col = build_B(hist_df, item_ids, args.lam)

    # ---- val 评估 ----
    hist_val = build_hist(hist_df)
    vs = np.load(f"{sc_dir}/val_sets.npz")
    v_src, v_t, v_c = vs["src"], vs["time"], vs["cands"]
    log(f"scoring val {v_c.shape}...")
    v_sc = score_cands(B, item2col, hist_val, v_src, v_t, v_c)
    mrr = mrr_expected(v_sc[:, 0], v_sc[:, 1:])
    log(f">>> EASE val 100-way MRR = {mrr:.4f}")
    np.savez_compressed(f"{sc_dir}/val_{args.tag}.npz", score=v_sc)

    # ---- test ----
    if args.score_test:
        log("rebuild B on full train for test...")
        B2, i2c2 = build_B(full_df, item_ids, args.lam)
        hist_full = build_hist(full_df)
        t_src = te.src.values; t_t = te.time.values
        t_c = te[cand_cols].values.astype(np.int64)
        log(f"scoring test {t_c.shape}...")
        t_sc = score_cands(B2, i2c2, hist_full, t_src, t_t, t_c)
        np.savez_compressed(f"{sc_dir}/test_{args.tag}.npz", score=t_sc)
        log(f"saved test_{args.tag}.npz")


if __name__ == "__main__":
    main()
