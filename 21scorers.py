"""
21scorers.py — 非神经打分器 + 诚实 100-way MRR 验证 harness

依据 EDA 的关键设计：
- val 负样本从【测试候选池】采样 + 注入用户历史 hard negative（匹配真实 test 分布）。
- 冻结 val 候选集到磁盘（col0=正样本），供 SASRec dumper 复用同一集合做融合。
- 防泄漏 + 正确性：
    val 打分只用 hist_df（split=0 / 时间前段）的统计；
    test 打分用【全量 train】的统计（test 在所有训练数据之后）。

打分器：repeat / pop / recent_pop / itemcf
用法：
  python 21scorers.py --dataset dataset1 --val_size 8000 --n_hard 2
  python 21scorers.py --dataset dataset2 --val_size 12000 --n_hard 2
"""
import os, os.path as osp, argparse, time
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.sparse as sp

RNG = np.random.RandomState(42)
DAY = 86400.0


def log(m):
    print(m, flush=True)


def mrr_100way(pos_scores, neg_scores):
    ranks = 1 + np.sum(neg_scores > pos_scores[:, None], axis=1)
    return float(np.mean(1.0 / ranks))


class Stats:
    """从一个历史 df 构建所有非神经统计。"""
    def __init__(self, df, item_max, hist_cap, cf_topk, tau):
        self.hist_cap = hist_cap
        self.tau = tau
        self.item_max = item_max
        # 用户历史
        h = defaultdict(list)
        for s, d, t in zip(df.src.values, df.dst.values, df.time.values):
            h[int(s)].append((float(t), int(d)))
        for s in h:
            h[s].sort()
        self.hist = h
        # pair 统计
        pc = defaultdict(int); pl = {}
        for s, d, t in zip(df.src.values, df.dst.values, df.time.values):
            k = (int(s), int(d)); pc[k] += 1; tt = float(t)
            if k not in pl or tt > pl[k]:
                pl[k] = tt
        self.pair_cnt = pc; self.pair_last = pl
        # 流行度
        self.dst_freq = df.dst.value_counts().to_dict()
        max_t = float(df.time.max())
        self.recent_freq = df[df.time >= max_t - 90 * DAY].dst.value_counts().to_dict()
        # ItemCF（scipy 稀疏）
        t0 = time.time()
        uids = list(h.keys()); u2i = {u: i for i, u in enumerate(uids)}
        rows, cols = [], []
        for u in uids:
            items = {d for _, d in h[u][-hist_cap:]}
            ui = u2i[u]
            for it in items:
                rows.append(ui); cols.append(it)
        M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(uids), item_max + 1))
        C = (M.T @ M).tocsr()
        ifu = np.asarray(M.sum(axis=0)).ravel()
        inv = np.zeros(item_max + 1, np.float32)
        nz = ifu > 0; inv[nz] = 1.0 / np.sqrt(ifu[nz])
        D = sp.diags(inv)
        S = (D @ C @ D).tocsr(); S.setdiag(0); S.eliminate_zeros()
        # top-k 行截断
        data, idx, indptr = [], [], [0]
        for r in range(S.shape[0]):
            seg = slice(S.indptr[r], S.indptr[r + 1])
            d = S.data[seg]; ix = S.indices[seg]
            if len(d) > cf_topk:
                sel = np.argpartition(d, -cf_topk)[-cf_topk:]
                d = d[sel]; ix = ix[sel]
            data.append(d); idx.append(ix); indptr.append(indptr[-1] + len(d))
        self.S = sp.csr_matrix(
            (np.concatenate(data) if data else np.zeros(0, np.float32),
             np.concatenate(idx) if idx else np.zeros(0, int),
             np.array(indptr)), shape=S.shape)
        log(f"  Stats built: itemcf nnz={self.S.nnz}, t={time.time()-t0:.1f}s")

    def profile(self, s, t):
        prof = {}
        for tt, d in self.hist.get(int(s), [])[-self.hist_cap:]:
            if tt >= t:
                continue
            prof[d] = prof.get(d, 0.0) + np.exp(-(t - tt) / self.tau)
        return prof

    def sc_repeat(self, s, t, cands):
        out = np.zeros(len(cands), np.float32)
        for j, c in enumerate(cands):
            k = (int(s), int(c)); cnt = self.pair_cnt.get(k, 0)
            if cnt > 0:
                lt = self.pair_last.get(k, 0.0)
                dec = np.exp(-(t - lt) / self.tau) if t > lt else 1.0
                out[j] = np.log1p(cnt) * dec
        return out

    def sc_pop(self, s, t, cands):
        return np.array([np.log1p(self.dst_freq.get(int(c), 0)) for c in cands], np.float32)

    def sc_recentpop(self, s, t, cands):
        return np.array([np.log1p(self.recent_freq.get(int(c), 0)) for c in cands], np.float32)

    def sc_itemcf(self, s, t, cands):
        prof = self.profile(s, t)
        out = np.zeros(len(cands), np.float32)
        if not prof:
            return out
        hist_items = set(prof.keys())
        S = self.S
        for j, c in enumerate(cands):
            c = int(c)
            if c > self.item_max:
                continue
            seg = slice(S.indptr[c], S.indptr[c + 1])
            nbr = S.indices[seg]; w = S.data[seg]
            acc = 0.0
            for kk in range(len(nbr)):
                hh = int(nbr[kk])
                if hh in hist_items:
                    acc += float(w[kk]) * prof[hh]
            out[j] = acc
        return out

    def scorers(self):
        return {"repeat": self.sc_repeat, "pop": self.sc_pop,
                "recent_pop": self.sc_recentpop, "itemcf": self.sc_itemcf}


def score_block(cand_mat, fn, srcs, times):
    out = np.zeros(cand_mat.shape, np.float32)
    for i in range(cand_mat.shape[0]):
        out[i] = fn(srcs[i], times[i], cand_mat[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--val_size", type=int, default=10000)
    ap.add_argument("--hist_cap", type=int, default=50)
    ap.add_argument("--cf_topk", type=int, default=20)
    ap.add_argument("--decay_tau_days", type=float, default=180.0)
    ap.add_argument("--n_hard", type=int, default=2)
    ap.add_argument("--new_edge_val", action="store_true",
                    help="val 正样本仅取新边（(src,dst) 不在 hist_df）以匹配 test 的归纳分布")
    ap.add_argument("--out_dir", default="./data/scorers")
    args = ap.parse_args()

    tau = args.decay_tau_days * DAY
    out_dir = osp.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    log(f"==== scorers for {args.dataset} ====")
    tr = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
    te = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")
    cand_cols = [c for c in te.columns if c.startswith("c")]

    if "split" in tr.columns:
        hist_df = tr[tr.split == 0].reset_index(drop=True)
        val_df = tr[tr.split == 1].reset_index(drop=True)
    else:
        tr = tr.sort_values("time", kind="stable").reset_index(drop=True)
        n_val = int(len(tr) * 0.15)
        hist_df = tr.iloc[:-n_val].reset_index(drop=True)
        val_df = tr.iloc[-n_val:].reset_index(drop=True)
    full_df = tr  # 全量训练（test 用）
    item_max = int(max(tr.dst.max(), te[cand_cols].values.max()))
    log(f"hist={len(hist_df)} val={len(val_df)} full={len(full_df)} item_max={item_max}")

    log("building VAL stats (hist_df)...")
    st_val = Stats(hist_df, item_max, args.hist_cap, args.cf_topk, tau)

    # ---- 构建 val 100-way 候选集 ----
    test_pool = np.unique(te[cand_cols].values.ravel()).astype(np.int64)
    val_pos_by_st = defaultdict(set)
    for s, d, t in zip(val_df.src.values, val_df.dst.values, val_df.time.values):
        val_pos_by_st[(int(s), float(t))].add(int(d))
    if args.new_edge_val:
        hist_pairs = set(zip(hist_df.src.values.tolist(), hist_df.dst.values.tolist()))
        mask = [(int(s), int(d)) not in hist_pairs
                for s, d in zip(val_df.src.values, val_df.dst.values)]
        val_df = val_df[pd.Series(mask, index=val_df.index)].reset_index(drop=True)
        log(f"new_edge_val: filtered to {len(val_df)} new-edge positives")
    n_val = min(args.val_size, len(val_df))
    sel = RNG.choice(len(val_df), size=n_val, replace=False)
    vsub = val_df.iloc[sel].reset_index(drop=True)
    val_src = vsub.src.values.astype(np.int64)
    val_time = vsub.time.values.astype(np.float64)
    val_pos = vsub.dst.values.astype(np.int64)
    val_cands = np.zeros((n_val, 100), dtype=np.int64)
    for i in range(n_val):
        s, t, pos = int(val_src[i]), float(val_time[i]), int(val_pos[i])
        exclude = val_pos_by_st[(s, t)]
        negs = []
        seq = st_val.hist.get(s, [])
        hi = list({d for _, d in seq if d != pos and d not in exclude})
        if hi and args.n_hard > 0:
            k = min(args.n_hard, len(hi))
            negs.extend(int(x) for x in RNG.choice(hi, size=k, replace=False))
        for x in RNG.choice(test_pool, size=140, replace=False):
            xi = int(x)
            if xi != pos and xi not in exclude and xi not in negs:
                negs.append(xi)
                if len(negs) >= 99:
                    break
        while len(negs) < 99:
            x = int(RNG.choice(test_pool))
            if x != pos and x not in exclude and x not in negs:
                negs.append(x)
        val_cands[i, 0] = pos
        val_cands[i, 1:] = negs[:99]
    log(f"val sets: {val_cands.shape} (n_hard={args.n_hard})")

    # ---- val 评估 ----
    results, val_dump = {}, {}
    for name, fn in st_val.scorers().items():
        t0 = time.time()
        sc = score_block(val_cands, fn, val_src, val_time)
        results[name] = mrr_100way(sc[:, 0], sc[:, 1:])
        val_dump[name] = sc
        log(f"  [{name}] val 100-way MRR = {results[name]:.4f}  ({time.time()-t0:.1f}s)")
    np.savez_compressed(f"{out_dir}/val_sets.npz", src=val_src, time=val_time, cands=val_cands)
    np.savez_compressed(f"{out_dir}/val_scores.npz", **val_dump)

    # ---- TEST 统计（全量 train）+ 打分 ----
    log("building TEST stats (full train)...")
    st_test = Stats(full_df, item_max, args.hist_cap, args.cf_topk, tau)
    te_src = te.src.values.astype(np.int64)
    te_time = te.time.values.astype(np.float64)
    te_cands = te[cand_cols].values.astype(np.int64)
    test_dump = {}
    for name, fn in st_test.scorers().items():
        t0 = time.time()
        test_dump[name] = score_block(te_cands, fn, te_src, te_time)
        log(f"  [{name}] test scored ({time.time()-t0:.1f}s)")
    np.savez_compressed(f"{out_dir}/test_scores.npz", **test_dump)

    log("\n==== summary (val 100-way MRR) ====")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        log(f"  {k:12s} {v:.4f}")
    log(f"saved → {out_dir}")


if __name__ == "__main__":
    main()
