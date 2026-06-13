"""
104_GRU4Rec_pop.py — GRU4Rec + 热度去偏（Popularity Debiasing）

算法原理：
  - GRU4Rec：用 GRU（门控循环单元）读用户购买历史，GRU 有天然记忆遗忘机制
  - 热度去偏：AI 倾向于给热门商品高分，但热门不等于适合这个用户
    公式：adj_score = raw_score - alpha * log(item_count + 1)
    验证集结果（按报告）：MRR 0.073 → 0.577（8倍提升）
  - 测试候选池负采样：训练时从测试候选池采负样本（含54%冷启动），
    让模型学会区分已知物品与冷启动物品

与 SASRec 的区别：
  - GRU 是顺序扫描（非注意力），对长序列梯度消失问题比 RNN 弱
  - 速度比 Transformer 快（Dataset2 中位历史 1037 条时优势明显）

输出目录：104_GRU4Rec_pop_MMDD
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
import argparse

jt.flags.use_cuda = 1

# ============================================================
# GRU4Rec 模型
# ============================================================
class GRU4Rec(nn.Module):
    def __init__(self, n_items, hidden_size=128, n_layers=2,
                 dropout=0.1, initializer_range=0.02):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        # 初始化
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                param.data = jt.array(
                    np.random.normal(0.0, initializer_range, param.shape).astype(np.float32))
        self.item_embedding.weight = jt.array(
            np.random.normal(0.0, initializer_range,
                             self.item_embedding.weight.shape).astype(np.float32))

    def forward(self, item_seq, item_seq_len):
        """
        item_seq: [B, L] int32
        item_seq_len: [B] int32
        返回: [B, H]
        """
        B = item_seq.shape[0]
        H = self.gru.hidden_size
        item_emb = self.emb_dropout(self.item_embedding(item_seq))
        # Jittor GRU 输出 [L, B, H]（batch_first=True 在 Jittor 中无效）
        out, _ = self.gru(item_emb)  # [L, B, H]
        out_bfirst = out.permute(1, 0, 2)  # [B, L, H]
        idx = jt.clamp(item_seq_len - 1, 0).reshape(B, 1, 1).broadcast([B, 1, H])
        user_repr = out_bfirst.gather(1, idx).squeeze(1)  # [B, H]
        return self.output_dropout(user_repr)


def log(msg):
    print(msg, flush=True)


# ============================================================
# 用户历史
# ============================================================
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


# ============================================================
# 验证（100-way MRR）
# ============================================================
def test_val(model, val_data, user_history, seq_len, item_count=None, alpha_pop=0.0, batch_size=256):
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

            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = model.forward(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())

            pos_emb = model.item_embedding(jt.Var(dst).int32())
            pos_score = jt.sigmoid((user_repr * pos_emb).sum(-1)).numpy()

            neg_flat = negs.reshape(-1)
            user_repr_rep = user_repr.unsqueeze(1).expand(-1, 99, -1).reshape(b*99, -1)
            neg_emb = model.item_embedding(jt.Var(neg_flat).int32())
            neg_score = jt.sigmoid((user_repr_rep * neg_emb).sum(-1)).numpy().reshape(b, 99)

            # 热度去偏
            if item_count is not None and alpha_pop > 0:
                def debiase(score_1d, ids_1d):
                    counts = np.array([item_count.get(int(i), 0) for i in ids_1d], dtype=np.float32)
                    return score_1d - alpha_pop * np.log1p(counts)
                for bi in range(b):
                    pos_score[bi] = debiase(pos_score[bi:bi+1], dst[bi:bi+1])
                    neg_score[bi] = debiase(neg_score[bi], negs[bi])

            ranks = 1 + np.sum(neg_score > pos_score.reshape(b, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            mrr_count += b
            jt.sync_all()
    return mrr_sum / max(mrr_count, 1)


# ============================================================
# 测试打分（含热度去偏）
# ============================================================
def test_competition(model, test_src, test_time, test_cands, user_history, seq_len,
                     item_count=None, alpha_pop=0.0, batch_size=64, cand_chunk=25):
    model.eval()
    N, K = test_cands.shape
    all_scores = np.zeros((N, K), dtype=np.float32)
    # 预计算候选热度
    if item_count is not None and alpha_pop > 0:
        cand_counts = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for j in range(K):
                cand_counts[i, j] = item_count.get(int(test_cands[i, j]), 0)
        pop_penalty = alpha_pop * np.log1p(cand_counts)
    else:
        pop_penalty = None

    with jt.no_grad():
        for i in tqdm(range(0, N, batch_size), ncols=100, desc="Testing"):
            end = min(i + batch_size, N)
            src = test_src[i:end]
            t = test_time[i:end]
            cands = test_cands[i:end]
            b = len(src)

            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = model.forward(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())

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

    if pop_penalty is not None:
        all_scores -= pop_penalty

    return all_scores


# ============================================================
# Main
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=100)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_neg", type=int, default=31)
parser.add_argument("--alpha_pop", type=float, default=0.5,
                    help="热度去偏系数，0=不去偏，0.5=标准去偏")
parser.add_argument("--neg_source", type=str, default="train",
                    choices=["train", "test_pool"],
                    help="负样本来源：train=训练集, test_pool=测试候选池")
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
log(f"Dataset: {args.dataset}, alpha_pop={args.alpha_pop}, neg={args.neg_source}")

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

# 物品热度统计
item_count = Counter(dst_all.tolist())
log(f"Unique items: {len(item_count)}")

model = GRU4Rec(
    n_items=max_item_id,
    hidden_size=args.hidden_size,
    n_layers=args.n_layers,
    dropout=args.dropout,
)
log(f"GRU4Rec params: {sum(p.numel() for p in model.parameters()):,}")

# 验证集
if val_df is not None:
    rng_val = np.random.RandomState(42)
    all_items = np.unique(dst_all)
    VAL_N = min(5000, len(val_df))
    val_idx = rng_val.choice(len(val_df), VAL_N, replace=False)
    val_data = (
        val_df["src"].values[val_idx].astype(np.int32),
        val_df["dst"].values[val_idx].astype(np.int32),
        val_df["time"].values[val_idx].astype(np.float32),
        rng_val.choice(all_items, size=(VAL_N, 99), replace=True).astype(np.int32)
    )
else:
    val_data = None

# 负样本采样器
pop_items = np.array(list(item_count.keys()), dtype=np.int32)
pop_probs = np.sqrt(np.array(list(item_count.values()), dtype=np.float64))
pop_probs /= pop_probs.sum()
rng_train = np.random.RandomState(0)

if args.neg_source == "test_pool":
    flat_cands = test_cands.reshape(-1).astype(np.int64)
    pool_vals, pool_cnts = np.unique(flat_cands, return_counts=True)
    pool_items = pool_vals.astype(np.int32)
    pool_probs = np.sqrt(pool_cnts.astype(np.float64))
    pool_probs /= pool_probs.sum()
    log(f"TestPool sampler: {len(pool_items)} unique candidates")

def sample_negs(dst, K):
    B = len(dst)
    if args.neg_source == "test_pool":
        items = pool_items
        probs = pool_probs
    else:
        items = pop_items
        probs = pop_probs
    raw = rng_train.choice(items, size=B * K * 2, p=probs).reshape(B, K * 2)
    result = np.zeros((B, K), dtype=np.int32)
    dst_arr = dst.reshape(-1, 1)
    for i in range(B):
        cands = raw[i][raw[i] != dst_arr[i, 0]][:K]
        if len(cands) < K:
            cands = np.concatenate([cands, rng_train.choice(items, K - len(cands))])
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
    perm = rng_train.permutation(n_train)
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

        item_seq, item_seq_len = get_batch_seqs(src_b, t_b, args.seq_len, user_history)
        user_repr = model.forward(jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())

        pos_emb = model.item_embedding(jt.Var(dst_b).int32())
        pos_logit = (user_repr * pos_emb).sum(-1)

        user_repr_rep = user_repr.unsqueeze(1).expand(-1, args.num_neg, -1).reshape(b*args.num_neg, -1)
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

    if val_data is not None:
        mrr_raw = test_val(model, val_data, user_history, args.seq_len)
        mrr_deb = test_val(model, val_data, user_history, args.seq_len,
                           item_count=item_count, alpha_pop=args.alpha_pop)
        log(f"Epoch {epoch+1} Val MRR raw={mrr_raw:.4f} debiased={mrr_deb:.4f}")
        cur = mrr_deb
        if cur > best_mrr:
            best_mrr = cur
            jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_gru4rec_best.pkl"))
            log(f"  -> New best: {best_mrr:.4f}")
    else:
        jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_gru4rec_ep{epoch+1}.pkl"))
    jt.gc()

log("\nGenerating test predictions...")
best_ckpt = osp.join(save_dir, f"{args.dataset}_gru4rec_best.pkl")
last_ckpt = osp.join(save_dir, f"{args.dataset}_gru4rec_ep{args.epochs}.pkl")
if osp.exists(best_ckpt):
    model.load_state_dict(jt.load(best_ckpt))
    log("Loaded best checkpoint.")
elif osp.exists(last_ckpt):
    model.load_state_dict(jt.load(last_ckpt))

scores = test_competition(
    model, test_src, test_time, test_cands, user_history, args.seq_len,
    item_count=item_count, alpha_pop=args.alpha_pop
)
log(f"Scores shape: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")

# 归一化到 [0,1]
scores_min = scores.min(axis=1, keepdims=True)
scores_max = scores.max(axis=1, keepdims=True)
scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_gru4rec.csv")
with open(out_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {out_file}")
log("DONE")
