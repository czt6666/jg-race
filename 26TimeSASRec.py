"""
26TimeSASRec.py — 时间感知 SASRec（TiSASRec-lite）

动机（诊断证据）：SASRec 只用交互【顺序】，忽略真实【时间】。
  - 大时间间隔查询差（180-365天 MRR 0.38 vs <30天 0.49）
  - 长历史用户差（截断 + 无时间衰减）

改造：input_emb = item_emb + position_emb + **time_gap_emb**
  time_gap = query_time - item_time，对数分桶（0/1天/3/7/14/30/90/180/365/2年/更久）。
  让模型知道每个历史交互"距今多久"，从而对近期/久远赋予不同权重。

损失：全词表 softmax（实测最优）。
评估：诚实 100-way harness（21scorers 的 val_sets.npz）。
输出：val_{tag}.npz + test_{tag}.npz（可直接并入 23/集成）。

用法：
  python 26TimeSASRec.py --dataset dataset2 --seq_len 128 --tag timesas --score_test
"""
import os, os.path as osp, sys, argparse, time
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"
import numpy as np
import pandas as pd
import jittor as jt
from jittor import nn
from jittor_geometric.nn.models.sasrec import SASRec

CPU = os.environ.get("DUMP_CPU", "0") == "1"
jt.flags.use_cuda = 0 if CPU else 1

DAY = 86400.0
# 时间间隔分桶边界（天）→ 桶 0..len
GAP_BINS = np.array([0, 1, 3, 7, 14, 30, 90, 180, 365, 730, 1460], dtype=np.float64)
N_GAP = len(GAP_BINS) + 2  # 桶: 0=pad, 1..len+1=digitize+1 (digitize∈0..len)


def log(m): print(m, flush=True)


class TimeAwareSASRec(SASRec):
    """在 SASRec 基础上加时间间隔 embedding。"""
    def __init__(self, *a, n_gap=N_GAP, **kw):
        super().__init__(*a, **kw)
        self.time_embedding = nn.Embedding(n_gap, self.hidden_size)
        # 初始化为小值
        self.time_embedding.weight = jt.array(
            np.random.normal(0, self.initializer_range, self.time_embedding.weight.shape))

    def forward_t(self, item_seq, item_seq_len, gap_bucket):
        item_seq_len = item_seq_len.clone()
        item_seq_len[item_seq_len == 0] = 1
        pos_ids = jt.arange(item_seq.size(1), dtype=jt.int64).unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(pos_ids)
        item_emb = self.item_embedding(item_seq)
        time_emb = self.time_embedding(gap_bucket)
        input_emb = item_emb + pos_emb + time_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        mask = self.get_attention_mask(item_seq)
        out = self.trm_encoder(input_emb, mask, output_all_encoded_layers=True)[-1]
        return self.gather_indexes(out, item_seq_len - 1)


def build_user_history(src, dst, t):
    raw = {}
    for s, d, ts in zip(src, dst, t):
        u = int(s)
        if u not in raw: raw[u] = ([], [])
        raw[u][0].append(float(ts)); raw[u][1].append(int(d))
    out = {}
    for u, (ti, it) in raw.items():
        ti = np.array(ti, np.float64); it = np.array(it, np.int32)
        o = np.argsort(ti, kind="stable")
        out[u] = (ti[o], it[o])
    return out


def get_batch_seqs_time(src, t, seq_len, hist):
    """返回 item_seq[B,L], len[B], gap_bucket[B,L]（距 query 的时间桶）。"""
    B = len(src)
    seq = np.zeros((B, seq_len), np.int32)
    slen = np.ones(B, np.int32)
    gap = np.zeros((B, seq_len), np.int32)  # 0=padding
    for i in range(B):
        u = int(src[i]); qt = float(t[i])
        if u not in hist: continue
        ta, ia = hist[u]
        idx = int(np.searchsorted(ta, qt, side="left"))
        if idx == 0: continue
        s0 = max(0, idx - seq_len)
        items = ia[s0:idx]; times = ta[s0:idx]; n = len(items)
        seq[i, :n] = items
        slen[i] = n
        gd = (qt - times) / DAY  # 距今天数
        gb = np.digitize(gd, GAP_BINS).astype(np.int32) + 1  # 1..len+1
        gap[i, :n] = np.clip(gb, 1, N_GAP - 1)  # 防越界, 0留给pad
    return seq, slen, gap


def mrr_exp(pos, neg):
    r = 1 + np.sum(neg > pos[:, None], 1) + 0.5 * np.sum(neg == pos[:, None], 1)
    return float(np.mean(1.0 / r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--inner_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--early_stop", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--softmax_tau", type=float, default=1.0)
    ap.add_argument("--scorers_dir", default="./data/scorers")
    ap.add_argument("--save_dir", default="./saved_models")
    ap.add_argument("--tag", default="timesas")
    ap.add_argument("--score_test", action="store_true")
    args = ap.parse_args()

    sc_dir = osp.join(args.scorers_dir, args.dataset)
    df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
    te = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")
    cand_cols = [c for c in te.columns if c.startswith("c")]
    if "split" in df.columns:
        train_df = df[df.split == 0].reset_index(drop=True)
    else:
        df = df.sort_values("time", kind="stable").reset_index(drop=True)
        train_df = df.iloc[:-int(len(df) * 0.15)].reset_index(drop=True)
    max_item = int(max(df.dst.max(), te[cand_cols].values.max()))
    log(f"{args.dataset}: max_item={max_item}, train_edges={len(train_df)}, N_GAP={N_GAP}")

    # 用户历史（训练用 split0；test 用全量）
    hist_train = build_user_history(train_df.src.values, train_df.dst.values, train_df.time.values)

    model = TimeAwareSASRec(
        n_layers=args.n_layers, n_heads=args.n_heads, hidden_size=args.hidden_size,
        inner_size=args.inner_size, hidden_dropout_prob=args.dropout,
        attn_dropout_prob=args.dropout, hidden_act="gelu", layer_norm_eps=1e-12,
        initializer_range=0.02, n_items=max_item, max_seq_length=args.seq_len)
    model.set_min_idx(0, 0)
    log(f"TimeAwareSASRec params: {sum(p.numel() for p in model.parameters()):,}")
    opt = jt.nn.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # val sets（诚实 harness）
    vs = np.load(f"{sc_dir}/val_sets.npz")
    v_src, v_t, v_c = vs["src"], vs["time"], vs["cands"]

    def score_val(hist):
        model.eval()
        N, C = v_c.shape
        out = np.zeros((N, C), np.float32)
        with jt.no_grad():
            bs = 200
            for s in range(0, N, bs):
                e = min(s + bs, N); b = e - s
                seq, sl, gp = get_batch_seqs_time(v_src[s:e], v_t[s:e], args.seq_len, hist)
                ur = model.forward_t(jt.Var(seq).int32(), jt.Var(sl).int32(), jt.Var(gp).int32())
                for cs in range(0, C, 20):
                    ce = min(cs + 20, C); cw = ce - cs
                    ch = v_c[s:e, cs:ce]
                    urr = ur.unsqueeze(1).expand(-1, cw, -1).reshape(b * cw, -1)
                    emb = model.item_embedding(jt.Var(ch.reshape(-1).astype(np.int32)))
                    out[s:e, cs:ce] = jt.sigmoid((urr * emb).sum(-1)).numpy().reshape(b, cw)
        return out

    train_src = train_df.src.values.astype(np.int32)
    train_dst = train_df.dst.values.astype(np.int32)
    train_t = train_df.time.values.astype(np.float64)
    n = len(train_src)
    best = 0.0; patience = 0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = f"{args.save_dir}/{args.dataset}_{args.tag}_best.pkl"

    for ep in range(args.epochs):
        model.train()
        order = np.random.permutation(n)
        losses = []
        nb = (n + args.batch_size - 1) // args.batch_size
        for bi in range(nb):
            idx = order[bi * args.batch_size:(bi + 1) * args.batch_size]
            seq, sl, gp = get_batch_seqs_time(train_src[idx], train_t[idx], args.seq_len, hist_train)
            ur = model.forward_t(jt.Var(seq).int32(), jt.Var(sl).int32(), jt.Var(gp).int32())
            W = model.item_embedding.weight
            logits = jt.matmul(ur, W.transpose(0, 1)) / args.softmax_tau
            loss = jt.nn.cross_entropy_loss(logits, jt.Var(train_dst[idx]).int32())
            opt.zero_grad(); opt.step(loss); jt.sync_all()
            losses.append(loss.item())
            if (bi + 1) % 200 == 0:
                log(f"  ep{ep+1} {bi+1}/{nb} loss={np.mean(losses[-200:]):.4f}")
            if (bi + 1) % 16 == 0:
                jt.gc()
        mrr = mrr_exp(*(lambda s: (s[:, 0], s[:, 1:]))(score_val(hist_train)))
        log(f"Epoch {ep+1} loss={np.mean(losses):.4f}  honest val MRR={mrr:.4f}")
        if mrr > best:
            best = mrr; patience = 0
            jt.save(model.state_dict(), ckpt); os.sync()
            log(f"  -> new best {best:.4f}")
        else:
            patience += 1
            if patience >= args.early_stop:
                log("early stop"); break

    log(f"BEST honest val MRR = {best:.4f}")
    # 用最佳模型输出 val 分数
    model.load_state_dict(jt.load(ckpt))
    np.savez_compressed(f"{sc_dir}/val_{args.tag}.npz", score=score_val(hist_train))
    log(f"saved val_{args.tag}.npz")

    if args.score_test:
        hist_full = build_user_history(df.src.values, df.dst.values, df.time.values)
        t_src = te.src.values; t_t = te.time.values.astype(np.float64)
        t_c = te[cand_cols].values.astype(np.int64)
        N, C = t_c.shape
        out = np.zeros((N, C), np.float32)
        model.eval()
        with jt.no_grad():
            bs = 200
            for s in range(0, N, bs):
                e = min(s + bs, N); b = e - s
                seq, sl, gp = get_batch_seqs_time(t_src[s:e], t_t[s:e], args.seq_len, hist_full)
                ur = model.forward_t(jt.Var(seq).int32(), jt.Var(sl).int32(), jt.Var(gp).int32())
                for cs in range(0, C, 20):
                    ce = min(cs + 20, C); cw = ce - cs
                    ch = t_c[s:e, cs:ce]
                    urr = ur.unsqueeze(1).expand(-1, cw, -1).reshape(b * cw, -1)
                    emb = model.item_embedding(jt.Var(ch.reshape(-1).astype(np.int32)))
                    out[s:e, cs:ce] = jt.sigmoid((urr * emb).sum(-1)).numpy().reshape(b, cw)
                if (s // bs) % 50 == 0:
                    log(f"  test {e}/{N}")
        np.savez_compressed(f"{sc_dir}/test_{args.tag}.npz", score=out)
        log(f"saved test_{args.tag}.npz")


if __name__ == "__main__":
    main()
