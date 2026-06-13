"""
111_SASRec_100way_testpool.py — SASRec + 100-way InfoNCE + 测试候选池负样本

算法原理：
  - 普通 SASRec 用 K=15 的 InfoNCE（16-way）训练，和测试的 100-way 口径不一致
  - 本方法用 K=99 的 InfoNCE（100-way）：每次训练模拟真实比赛场景
  - 负样本来自测试候选池（含 54% 冷启动），让模型提前适应测试分布

两个关键改进（相比 13SASRec_wd.py）：
  1. K=99（100-way 训练），比 K=15 更接近测试难度
  2. 负样本来自测试候选池，覆盖冷启动物品（而非训练热门物品）

输出目录：111_SASRec_100way_testpool_MMDD
"""
import os
import os.path as osp
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from jittor_geometric.nn.models.sasrec import SASRec
import argparse

jt.flags.use_cuda = 1


def log(msg):
    print(msg, flush=True)


def build_user_history(src_arr, dst_arr, t_arr):
    raw = defaultdict(list)
    for s, d, ts in zip(src_arr, dst_arr, t_arr):
        raw[int(s)].append((float(ts), int(d)))
    result = {}
    for uid, lst in raw.items():
        lst.sort()
        result[uid] = (
            np.array([x[0] for x in lst], dtype=np.float32),
            np.array([x[1] for x in lst], dtype=np.int32),
        )
    return result


def get_batch_seqs(src, t, seq_len, user_history):
    B = len(src)
    item_seq = np.zeros((B, seq_len), dtype=np.int32)
    item_seq_len = np.ones(B, dtype=np.int32)
    for i in range(B):
        uid = int(src[i])
        qt = float(t[i])
        if uid not in user_history:
            continue
        ts_arr, it_arr = user_history[uid]
        idx = int(np.searchsorted(ts_arr, qt, side='left'))
        if idx == 0:
            continue
        start = max(0, idx - seq_len)
        items = it_arr[start:idx]
        n = len(items)
        item_seq[i, :n] = items
        item_seq_len[i] = n
    return item_seq, item_seq_len


def infonce_loss(pos_logit, neg_logit, K):
    B = pos_logit.shape[0]
    logits = jt.concat([pos_logit.unsqueeze(1), neg_logit.reshape(B, K)], dim=1)
    return jt.nn.cross_entropy_loss(logits, jt.zeros(B, dtype='int32'))


def test_val(backbone, val_data, user_history, seq_len, batch_size=256):
    backbone.eval()
    mrr_sum, mrr_count = 0.0, 0
    val_src, val_dst, val_t, val_negs = val_data
    with jt.no_grad():
        for i in range(0, len(val_src), batch_size):
            src = val_src[i:i+batch_size]
            dst = val_dst[i:i+batch_size]
            t = val_t[i:i+batch_size]
            negs = val_negs[i:i+batch_size]
            b = len(src)
            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = backbone.forward(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())
            pos_emb = backbone.item_embedding(jt.Var(dst).int32())
            pos_score = jt.sigmoid((user_repr * pos_emb).sum(-1)).numpy()
            neg_flat = negs.reshape(-1)
            user_repr_rep = user_repr.unsqueeze(1).expand(-1, 99, -1).reshape(b*99, -1)
            neg_emb = backbone.item_embedding(jt.Var(neg_flat).int32())
            neg_score = jt.sigmoid((user_repr_rep * neg_emb).sum(-1)).numpy().reshape(b, 99)
            ranks = 1 + np.sum(neg_score > pos_score.reshape(b, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            mrr_count += b
            jt.sync_all()
    return mrr_sum / max(mrr_count, 1)


def test_competition(backbone, test_src, test_time, test_cands, user_history, seq_len,
                     batch_size=64, cand_chunk=25):
    backbone.eval()
    N, K = test_cands.shape
    all_scores = np.zeros((N, K), dtype=np.float32)
    with jt.no_grad():
        for i in tqdm(range(0, N, batch_size), ncols=100, desc="Testing"):
            end = min(i + batch_size, N)
            src = test_src[i:end]
            t = test_time[i:end]
            cands = test_cands[i:end]
            b = len(src)
            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = backbone.forward(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())
            scores_buf = np.zeros((b, K), dtype=np.float32)
            for cs in range(0, K, cand_chunk):
                ce = min(cs + cand_chunk, K)
                chunk = cands[:, cs:ce]
                csz = ce - cs
                user_rep_exp = user_repr.unsqueeze(1).expand(-1, csz, -1).reshape(b*csz, -1)
                cand_emb = backbone.item_embedding(jt.Var(chunk.reshape(-1).astype(np.int32)))
                chunk_scores = jt.sigmoid((user_rep_exp * cand_emb).sum(-1)).numpy()
                scores_buf[:, cs:ce] = chunk_scores.reshape(b, csz)
                del user_rep_exp, cand_emb
                jt.sync_all()
            all_scores[i:end] = scores_buf
            if (i // batch_size + 1) % 50 == 0:
                jt.gc()
    return all_scores


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seq_len", type=int, default=100)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--use_all_train", action="store_true")
args = parser.parse_args()

date_str = datetime.now().strftime("%m%d")
script_name = osp.splitext(osp.basename(__file__))[0]
if args.output_dir is None:
    args.output_dir = f"./{script_name}_{date_str}"
save_dir = osp.join(args.output_dir, "ckpt")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(osp.join(args.output_dir, args.dataset), exist_ok=True)

log(f"=== {script_name} ===")
log(f"Dataset: {args.dataset}, 100-way InfoNCE, test-pool negatives")

df = pd.read_csv(osp.join(args.data_dir, args.dataset, "train.csv"))
test_df = pd.read_csv(osp.join(args.data_dir, args.dataset, "test.csv"))

src_all = df["src"].values.astype(np.int32)
dst_all = df["dst"].values.astype(np.int32)
t_all = df["time"].values.astype(np.float32)

if args.use_all_train:
    train_df = df
    val_df = None
elif "split" in df.columns:
    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df = df[df["split"] == 1].reset_index(drop=True)
else:
    n_val = int(len(df) * 0.1)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)

log(f"Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}")

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.float32)
test_cands = test_df.iloc[:, 2:].values.astype(np.int32)
user_history = build_user_history(src_all, dst_all, t_all)
max_item_id = max(int(dst_all.max()), int(test_cands.max()))
log(f"Max item ID: {max_item_id}")

backbone = SASRec(
    n_layers=args.n_layers, n_heads=args.n_heads,
    hidden_size=args.hidden_size, inner_size=args.hidden_size * 2,
    hidden_dropout_prob=args.dropout, attn_dropout_prob=args.dropout,
    hidden_act='gelu', layer_norm_eps=1e-12,
    initializer_range=0.02, n_items=max_item_id,
    max_seq_length=args.seq_len,
)
backbone.set_min_idx(0, 0)
log(f"SASRec params: {sum(p.numel() for p in backbone.parameters()):,}")

# 测试候选池负采样
flat_cands = test_cands.reshape(-1).astype(np.int64)
pool_vals, pool_cnts = np.unique(flat_cands, return_counts=True)
pool_items = pool_vals.astype(np.int32)
pool_probs = np.sqrt(pool_cnts.astype(np.float64))
pool_probs /= pool_probs.sum()
rng = np.random.RandomState(42)
log(f"Test pool: {len(pool_items)} unique candidates (includes cold-start items)")

def sample_100way_negs(dst, K=99):
    B = len(dst)
    raw = rng.choice(pool_items, size=B * K * 2, p=pool_probs).reshape(B, K * 2)
    result = np.zeros((B, K), dtype=np.int32)
    dst_arr = dst.reshape(-1, 1)
    for i in range(B):
        cands = raw[i][raw[i] != dst_arr[i, 0]][:K]
        if len(cands) < K:
            cands = np.concatenate([cands, rng.choice(pool_items, K-len(cands))])
        result[i] = cands[:K]
    return result.reshape(-1)

if val_df is not None:
    all_items = np.unique(dst_all)
    VAL_N = min(5000, len(val_df))
    val_idx = rng.choice(len(val_df), VAL_N, replace=False)
    val_data = (
        val_df["src"].values[val_idx].astype(np.int32),
        val_df["dst"].values[val_idx].astype(np.int32),
        val_df["time"].values[val_idx].astype(np.float32),
        rng.choice(all_items, size=(VAL_N, 99), replace=True).astype(np.int32)
    )
else:
    val_data = None

optimizer = jt.nn.Adam(list(backbone.parameters()), lr=args.lr, weight_decay=args.weight_decay)
best_mrr = 0.0
train_src_arr = train_df["src"].values.astype(np.int32)
train_dst_arr = train_df["dst"].values.astype(np.int32)
train_t_arr = train_df["time"].values.astype(np.float32)
n_train = len(train_df)
K = 99  # 100-way

for epoch in range(args.epochs):
    backbone.train()
    losses = []
    perm = rng.permutation(n_train)
    ep_src = train_src_arr[perm]
    ep_dst = train_dst_arr[perm]
    ep_t = train_t_arr[perm]

    pbar = tqdm(range(0, n_train, args.batch_size), ncols=120, desc=f"Epoch {epoch+1}")
    for step, start in enumerate(pbar):
        end = min(start + args.batch_size, n_train)
        src_b = ep_src[start:end]
        dst_b = ep_dst[start:end]
        t_b = ep_t[start:end]
        b = len(src_b)

        neg_b = sample_100way_negs(dst_b, K)

        item_seq, item_seq_len = get_batch_seqs(src_b, t_b, args.seq_len, user_history)
        user_repr = backbone.forward(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())

        pos_emb = backbone.item_embedding(jt.Var(dst_b).int32())
        pos_logit = (user_repr * pos_emb).sum(-1)

        user_repr_rep = user_repr.unsqueeze(1).expand(-1, K, -1).reshape(b * K, -1)
        neg_emb = backbone.item_embedding(jt.Var(neg_b).int32())
        neg_logit = (user_repr_rep * neg_emb).sum(-1)

        loss = infonce_loss(pos_logit, neg_logit, K)
        optimizer.zero_grad()
        optimizer.step(loss)
        jt.sync_all()

        losses.append(float(loss.item()))
        pbar.set_description(f"Epoch {epoch+1} loss={np.mean(losses[-50:]):.4f}")
        if (step + 1) % 32 == 0:
            jt.gc()

    log(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")

    if val_data is not None:
        mrr = test_val(backbone, val_data, user_history, args.seq_len)
        log(f"Epoch {epoch+1} Val MRR: {mrr:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            jt.save(backbone.state_dict(), osp.join(save_dir, f"{args.dataset}_100way_best.pkl"))
            log(f"  -> New best: {best_mrr:.4f}")
    else:
        jt.save(backbone.state_dict(), osp.join(save_dir, f"{args.dataset}_100way_ep{epoch+1}.pkl"))
    jt.gc()

log("\nGenerating predictions...")
best_ckpt = osp.join(save_dir, f"{args.dataset}_100way_best.pkl")
if osp.exists(best_ckpt):
    backbone.load_state_dict(jt.load(best_ckpt))

scores = test_competition(backbone, test_src, test_time, test_cands, user_history, args.seq_len)
log(f"Scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_100way.csv")
with open(out_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {out_file}")
log("DONE")
