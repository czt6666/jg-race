"""
22sasrec_dump.py — 用训练好的 SASRec 给【冻结的 val 候选集】+ test 候选逐候选打分，
                   输出对齐的分数 npz 以供集成（21scorers.py 生成 val_sets.npz）。

历史构建：从全量 train 构建用户序列，按 query 时间过滤（get_batch_seqs），
         与 13SASRec_wd.py 的 val/test 逻辑一致。

用法：
  python 22sasrec_dump.py --dataset dataset1 \
      --model_path saved_models/run_0604/17sasrec_dataset1_val/dataset1_SASRec_best.pkl \
      --seq_len 64 --score_test
"""
import os, os.path as osp, sys, argparse
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"
import numpy as np
import pandas as pd
import jittor as jt
from jittor_geometric.nn.models.sasrec import SASRec

CPU = os.environ.get("DUMP_CPU", "0") == "1"
jt.flags.use_cuda = 0 if CPU else 1


def log(m):
    print(m, flush=True)


def build_user_history(src, dst, t):
    raw = {}
    for s, d, ts in zip(src, dst, t):
        u = int(s)
        if u not in raw:
            raw[u] = ([], [])
        raw[u][0].append(float(ts)); raw[u][1].append(int(d))
    out = {}
    for u, (ti, it) in raw.items():
        ti = np.array(ti, np.float32); it = np.array(it, np.int32)
        o = np.argsort(ti, kind="stable")
        out[u] = (ti[o], it[o])
    return out


def get_batch_seqs(src, t, seq_len, hist):
    B = len(src)
    seq = np.zeros((B, seq_len), np.int32)
    slen = np.ones(B, np.int32)
    for i in range(B):
        u = int(src[i]); qt = float(t[i])
        if u not in hist:
            continue
        ta, ia = hist[u]
        idx = int(np.searchsorted(ta, qt, side="left"))
        if idx == 0:
            continue
        s0 = max(0, idx - seq_len)
        items = ia[s0:idx]; n = len(items)
        seq[i, :n] = items; slen[i] = n
    return seq, slen


def score_cands(backbone, src, t, cands, hist, seq_len, batch=200, chunk=20):
    """cands [N,100] → scores [N,100] (sigmoid dot)."""
    N, C = cands.shape
    out = np.zeros((N, C), np.float32)
    nb = (N + batch - 1) // batch
    with jt.no_grad():
        for bi in range(nb):
            s = bi * batch; e = min(s + batch, N)
            b = e - s
            seq, slen = get_batch_seqs(src[s:e], t[s:e], seq_len, hist)
            ur = backbone.forward(jt.Var(seq).int32(), jt.Var(slen).int32())  # [b,H]
            cb = cands[s:e]
            for cs in range(0, C, chunk):
                ce = min(cs + chunk, C); cw = ce - cs
                ch = cb[:, cs:ce]
                urr = ur.unsqueeze(1).expand(-1, cw, -1).reshape(b * cw, -1)
                emb = backbone.item_embedding(jt.Var(ch.reshape(-1).astype(np.int32)))
                sc = jt.sigmoid((urr * emb).sum(-1)).numpy().reshape(b, cw)
                out[s:e, cs:ce] = sc
                del urr, emb
            if (bi + 1) % 50 == 0:
                jt.gc()
            if (bi + 1) % 20 == 0:
                log(f"  scored {e}/{N}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--inner_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--scorers_dir", default="./data/scorers")
    ap.add_argument("--tag", default="sasrec")
    ap.add_argument("--score_test", action="store_true")
    args = ap.parse_args()

    sc_dir = osp.join(args.scorers_dir, args.dataset)
    tr = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
    te = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")
    cand_cols = [c for c in te.columns if c.startswith("c")]
    max_item = int(max(tr.dst.max(), te[cand_cols].values.max()))
    log(f"{args.dataset}: max_item={max_item}, building history from full train...")
    hist = build_user_history(tr.src.values, tr.dst.values, tr.time.values)

    backbone = SASRec(n_layers=args.n_layers, n_heads=args.n_heads,
                      hidden_size=args.hidden_size, inner_size=args.inner_size,
                      hidden_dropout_prob=args.dropout, attn_dropout_prob=args.dropout,
                      hidden_act="gelu", layer_norm_eps=1e-12, initializer_range=0.02,
                      n_items=max_item, max_seq_length=args.seq_len)
    backbone.set_min_idx(user_min_idx=0, item_min_idx=0)
    backbone.load_state_dict(jt.load(args.model_path))
    backbone.eval()
    log(f"loaded {args.model_path}")

    # ---- val ----
    vs = np.load(f"{sc_dir}/val_sets.npz")
    v_src, v_t, v_cands = vs["src"], vs["time"], vs["cands"]
    log(f"scoring val {v_cands.shape}...")
    v_sc = score_cands(backbone, v_src, v_t, v_cands, hist, args.seq_len)
    # 报告 val MRR（期望排名平局处理）
    pos = v_sc[:, 0]; neg = v_sc[:, 1:]
    ranks = 1 + np.sum(neg > pos[:, None], axis=1) + 0.5 * np.sum(neg == pos[:, None], axis=1)
    log(f"  val 100-way MRR ({args.tag}) = {np.mean(1.0/ranks):.4f}")
    np.savez_compressed(f"{sc_dir}/val_{args.tag}.npz", score=v_sc)
    log(f"  saved {sc_dir}/val_{args.tag}.npz")

    # ---- test ----
    if args.score_test:
        t_src = te.src.values.astype(np.int64)
        t_t = te.time.values.astype(np.float64)
        t_cands = te[cand_cols].values.astype(np.int64)
        log(f"scoring test {t_cands.shape}...")
        t_sc = score_cands(backbone, t_src, t_t, t_cands, hist, args.seq_len)
        np.savez_compressed(f"{sc_dir}/test_{args.tag}.npz", score=t_sc)
        log(f"  saved {sc_dir}/test_{args.tag}.npz")


if __name__ == "__main__":
    main()
