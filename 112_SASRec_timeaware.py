"""
112_SASRec_timeaware.py — SASRec + 时间间隔感知 Attention（Time-Gap Encoding）

算法原理：
  - 普通 SASRec 只用位置编码（position=0,1,2,...）来区分序列中的物品
  - 但「昨天看的」和「两年前看的」应该有不同权重！
  - 本方法：给 Attention Score 加上时间间隔惩罚
    attn_score = (Q @ K^T) / sqrt(d) - lambda * log(1 + time_gap_days)
  - 这样时间越远的物品注意力权重越低，最近的行为更被重视

实现方式：
  - 计算序列中每两个位置之间的时间间隔（天数）
  - 在每层 Transformer 的 attention 前加上 time_gap_bias 矩阵
  - time_gap_bias[i, j] = -lambda * log(1 + |t_i - t_j| / 86400)

输出目录：112_SASRec_timeaware_MMDD
"""
import os
import os.path as osp
import sys
import math
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
import argparse

jt.flags.use_cuda = 1


def log(msg):
    print(msg, flush=True)


class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, time_lambda=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.time_lambda = time_lambda
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def execute(self, x, attn_mask, time_gap_days=None):
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim
        Q = self.q_proj(x).reshape(B, L, H, D).permute(0, 2, 1, 3)  # [B,H,L,D]
        K = self.k_proj(x).reshape(B, L, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, L, H, D).permute(0, 2, 1, 3)
        scores = jt.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [B,H,L,L]
        if time_gap_days is not None:
            time_bias = -self.time_lambda * jt.log(1.0 + time_gap_days)  # [B,L,L]
            scores = scores + time_bias.unsqueeze(1)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)  # [B,1,L,L] -> broadcast to [B,H,L,L]
        attn_weights = jt.nn.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = jt.matmul(attn_weights, V)  # [B,H,L,D]
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.out_proj(out)

    def forward(self, x, attn_mask, time_gap_days=None):
        return self.execute(x, attn_mask, time_gap_days)


class TimeAwareSASRecLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, time_lambda=0.1):
        super().__init__()
        self.attn = TimeAwareMultiHeadAttention(hidden_size, n_heads, dropout, time_lambda)
        self.ffn1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ffn2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def execute(self, x, attn_mask, time_gap_days=None):
        res = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask, time_gap_days)
        x = self.dropout(x) + res
        res = x
        x = self.norm2(x)
        x = self.dropout(self.ffn2(self.act(self.ffn1(x)))) + res
        return x

    def forward(self, x, attn_mask, time_gap_days=None):
        return self.execute(x, attn_mask, time_gap_days)


class TimeAwareSASRec(nn.Module):
    def __init__(self, n_items, seq_len, hidden_size, n_layers, n_heads, dropout,
                 time_lambda=0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 2, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_len + 1, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TimeAwareSASRecLayer(hidden_size, n_heads, dropout, time_lambda)
            for _ in range(n_layers)
        ])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        jt.init.gauss_(self.item_embedding.weight, 0, 0.02)
        jt.init.gauss_(self.pos_embedding.weight, 0, 0.02)

    def get_attn_mask(self, item_seq):
        B, L = item_seq.shape
        # Causal mask: position i can attend to positions j <= i
        rows = jt.arange(L, dtype='float32').unsqueeze(1)  # [L, 1]
        cols = jt.arange(L, dtype='float32').unsqueeze(0)  # [1, L]
        causal = (rows >= cols).float()  # [L, L]: 1 where valid
        # Padding mask
        pad_valid = (item_seq != 0).float()  # [B, L]: 1 where non-padding
        # Combined: [B, L, L]
        combined = causal.unsqueeze(0) * pad_valid.unsqueeze(1)
        # Convert to additive bias: 0 valid, -10000 masked
        return (1.0 - combined) * -10000.0

    def execute(self, item_seq, item_seq_len, time_gap_days=None):
        B, L = item_seq.shape
        pos_ids = jt.arange(1, L + 1, dtype='int32').unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.pos_embedding(pos_ids)
        x = self.dropout(self.norm(x))
        mask = self.get_attn_mask(item_seq)
        for layer in self.layers:
            x = layer(x, mask, time_gap_days)
        # Gather the representation at the last non-padding position
        idx = (item_seq_len - 1).clamp(min_v=0).int32()  # [B]
        out = x[jt.arange(B), idx]  # [B, H]
        return out

    def forward(self, item_seq, item_seq_len, time_gap_days=None):
        return self.execute(item_seq, item_seq_len, time_gap_days)


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


def get_batch_seqs_with_time(src, t, seq_len, user_history):
    B = len(src)
    item_seq = np.zeros((B, seq_len), dtype=np.int32)
    time_seq = np.zeros((B, seq_len), dtype=np.float32)
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
        times = ts_arr[start:idx]
        n = len(items)
        item_seq[i, :n] = items
        time_seq[i, :n] = times
        item_seq_len[i] = n
    return item_seq, time_seq, item_seq_len


def compute_time_gap_matrix(time_seq, query_t):
    """
    time_seq: [B, L] float32 (unix timestamps, 0 for padding)
    query_t: [B] float32 (query timestamp)
    Returns: [B, L, L] float32 time gap in days
    """
    B, L = time_seq.shape
    # For each pair (i, j), time_gap[i, j] = |time[i] - time[j]| / 86400
    t = time_seq  # [B, L]
    t_i = t[:, :, np.newaxis]  # [B, L, 1]
    t_j = t[:, np.newaxis, :]  # [B, 1, L]
    gap_days = np.abs(t_i - t_j) / 86400.0
    # For padding positions (time=0), set gap to 0 (will be masked by attn_mask anyway)
    return gap_days.astype(np.float32)


def infonce_loss(pos_logit, neg_logit, K):
    B = pos_logit.shape[0]
    logits = jt.concat([pos_logit.unsqueeze(1), neg_logit.reshape(B, K)], dim=1)
    return jt.nn.cross_entropy_loss(logits, jt.zeros(B, dtype='int32'))


def test_val(model, val_data, user_history, seq_len, batch_size=256):
    model.eval()
    mrr_sum, mrr_count = 0.0, 0
    val_src, val_dst, val_t, val_negs = val_data
    with jt.no_grad():
        for i in range(0, len(val_src), batch_size):
            src = val_src[i:i+batch_size]
            dst = val_dst[i:i+batch_size]
            t = val_t[i:i+batch_size]
            negs = val_negs[i:i+batch_size]
            b = len(src)
            item_seq, time_seq, item_seq_len = get_batch_seqs_with_time(src, t, seq_len, user_history)
            gap_days = compute_time_gap_matrix(time_seq, t)
            user_repr = model(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32(),
                              jt.Var(gap_days))
            pos_emb = model.item_embedding(jt.Var(dst).int32())
            pos_score = jt.sigmoid((user_repr * pos_emb).sum(-1)).numpy()
            neg_flat = negs.reshape(-1)
            user_repr_rep = user_repr.unsqueeze(1).expand(-1, 99, -1).reshape(b*99, -1)
            neg_emb = model.item_embedding(jt.Var(neg_flat).int32())
            neg_score = jt.sigmoid((user_repr_rep * neg_emb).sum(-1)).numpy().reshape(b, 99)
            ranks = 1 + np.sum(neg_score > pos_score.reshape(b, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            mrr_count += b
            jt.sync_all()
    return mrr_sum / max(mrr_count, 1)


def test_competition(model, test_src, test_time, test_cands, user_history, seq_len,
                     batch_size=64, cand_chunk=25):
    model.eval()
    N, K = test_cands.shape
    all_scores = np.zeros((N, K), dtype=np.float32)
    with jt.no_grad():
        for i in tqdm(range(0, N, batch_size), ncols=100, desc="Testing"):
            end = min(i + batch_size, N)
            src = test_src[i:end]
            t = test_time[i:end]
            cands = test_cands[i:end]
            b = len(src)
            item_seq, time_seq, item_seq_len = get_batch_seqs_with_time(src, t, seq_len, user_history)
            gap_days = compute_time_gap_matrix(time_seq, t)
            user_repr = model(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32(),
                              jt.Var(gap_days))
            scores_buf = np.zeros((b, K), dtype=np.float32)
            for cs in range(0, K, cand_chunk):
                ce = min(cs + cand_chunk, K)
                chunk = cands[:, cs:ce]
                csz = ce - cs
                user_rep_exp = user_repr.unsqueeze(1).expand(-1, csz, -1).reshape(b*csz, -1)
                cand_emb = model.item_embedding(jt.Var(chunk.reshape(-1).astype(np.int32)))
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
parser.add_argument("--time_lambda", type=float, default=0.1)
parser.add_argument("--num_neg", type=int, default=31)
args = parser.parse_args()

date_str = datetime.now().strftime("%m%d")
script_name = osp.splitext(osp.basename(__file__))[0]
if args.output_dir is None:
    args.output_dir = f"./{script_name}_{date_str}"
save_dir = osp.join(args.output_dir, "ckpt")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(osp.join(args.output_dir, args.dataset), exist_ok=True)

log(f"=== {script_name} ===")
log(f"Dataset: {args.dataset}, time-gap attention lambda={args.time_lambda}")

df = pd.read_csv(osp.join(args.data_dir, args.dataset, "train.csv"))
test_df = pd.read_csv(osp.join(args.data_dir, args.dataset, "test.csv"))
src_all = df["src"].values.astype(np.int32)
dst_all = df["dst"].values.astype(np.int32)
t_all = df["time"].values.astype(np.float32)

if "split" in df.columns:
    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df = df[df["split"] == 1].reset_index(drop=True)
else:
    n_val = int(len(df) * 0.1)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)

log(f"Train: {len(train_df)}, Val: {len(val_df)}")
test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.float32)
test_cands = test_df.iloc[:, 2:].values.astype(np.int32)
user_history = build_user_history(src_all, dst_all, t_all)
max_item_id = max(int(dst_all.max()), int(test_cands.max()))
log(f"Max item ID: {max_item_id}")

model = TimeAwareSASRec(
    n_items=max_item_id, seq_len=args.seq_len,
    hidden_size=args.hidden_size, n_layers=args.n_layers,
    n_heads=args.n_heads, dropout=args.dropout,
    time_lambda=args.time_lambda,
)
log(f"TimeAwareSASRec params: {sum(p.numel() for p in model.parameters()):,}")

all_items = np.unique(dst_all)
rng = np.random.RandomState(42)
VAL_N = min(5000, len(val_df))
val_idx = rng.choice(len(val_df), VAL_N, replace=False)
val_data = (
    val_df["src"].values[val_idx].astype(np.int32),
    val_df["dst"].values[val_idx].astype(np.int32),
    val_df["time"].values[val_idx].astype(np.float32),
    rng.choice(all_items, size=(VAL_N, 99), replace=True).astype(np.int32)
)

dst_counter = Counter(train_df["dst"].values.tolist())
pop_items = np.array(list(dst_counter.keys()), dtype=np.int32)
pop_probs = np.sqrt(np.array(list(dst_counter.values()), dtype=np.float64))
pop_probs /= pop_probs.sum()

def sample_negs(dst, K):
    B = len(dst)
    raw = rng.choice(pop_items, size=B * K * 2, p=pop_probs).reshape(B, K * 2)
    result = np.zeros((B, K), dtype=np.int32)
    for i in range(B):
        cands = raw[i][raw[i] != dst[i]][:K]
        if len(cands) < K:
            cands = np.concatenate([cands, rng.choice(pop_items, K-len(cands))])
        result[i] = cands[:K]
    return result.reshape(-1)

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
best_mrr = 0.0
train_src_arr = train_df["src"].values.astype(np.int32)
train_dst_arr = train_df["dst"].values.astype(np.int32)
train_t_arr = train_df["time"].values.astype(np.float32)
n_train = len(train_df)

for epoch in range(args.epochs):
    model.train()
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
        neg_b = sample_negs(dst_b, args.num_neg)
        item_seq, time_seq, item_seq_len = get_batch_seqs_with_time(src_b, t_b, args.seq_len, user_history)
        gap_days = compute_time_gap_matrix(time_seq, t_b)
        user_repr = model(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32(),
                          jt.Var(gap_days))
        pos_emb = model.item_embedding(jt.Var(dst_b).int32())
        pos_logit = (user_repr * pos_emb).sum(-1)
        user_repr_rep = user_repr.unsqueeze(1).expand(-1, args.num_neg, -1).reshape(b * args.num_neg, -1)
        neg_emb = model.item_embedding(jt.Var(neg_b).int32())
        neg_logit = (user_repr_rep * neg_emb).sum(-1)
        loss = infonce_loss(pos_logit, neg_logit, args.num_neg)
        optimizer.zero_grad()
        optimizer.step(loss)
        jt.sync_all()
        losses.append(float(loss.item()))
        pbar.set_description(f"Epoch {epoch+1} loss={np.mean(losses[-50:]):.4f}")
        if (step + 1) % 32 == 0:
            jt.gc()

    log(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")
    mrr = test_val(model, val_data, user_history, args.seq_len)
    log(f"Epoch {epoch+1} Val MRR: {mrr:.4f}")
    if mrr > best_mrr:
        best_mrr = mrr
        jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_timeaware_best.pkl"))
        log(f"  -> New best: {best_mrr:.4f}")
    jt.gc()

log("\nGenerating predictions...")
best_ckpt = osp.join(save_dir, f"{args.dataset}_timeaware_best.pkl")
if osp.exists(best_ckpt):
    model.load_state_dict(jt.load(best_ckpt))

scores = test_competition(model, test_src, test_time, test_cands, user_history, args.seq_len)
log(f"Scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_timeaware.csv")
with open(out_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {out_file}")
log("DONE")
