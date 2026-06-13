"""
103_BERT4Rec_base.py — BERT4Rec (Bidirectional Encoder for Recommender)

算法原理：
  - SASRec 是从左到右的单向注意力（看历史预测未来）
  - BERT4Rec 用双向注意力 + 掩码语言模型（MLM）训练
  - 训练时：随机 mask 20% 的历史物品，让模型从两边猜中间
  - 推理时：在序列末尾加一个 [MASK] token，让模型预测"下一个"
  - 优势：双向上下文让模型对用户兴趣理解更全面

与 SASRec 的区别：
  1. attention mask 改为双向（不是 causal/lower-triangular）
  2. 训练损失：MLM CE（预测 mask 位置）vs SASRec 的 InfoNCE
  3. 推理时用 [MASK] token 位置的输出向量和候选做点积
  4. 物品 0 = padding，物品 n_items+1 = [MASK]

输出目录：103_BERT4Rec_base_MMDD
"""
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from datetime import datetime

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from jittor_geometric.nn.models.transformers import TransformerEncoderbyHand
import argparse

jt.flags.use_cuda = 1

# ============================================================
# BERT4Rec 模型
# ============================================================
class BERT4Rec(jt.nn.Module):
    def __init__(self, n_items, max_seq_len, hidden_size=128, inner_size=256,
                 n_layers=2, n_heads=2, dropout=0.1, layer_norm_eps=1e-12,
                 initializer_range=0.02):
        super().__init__()
        self.n_items = n_items
        self.mask_token = n_items + 1          # [MASK] id
        self.n_vocab = n_items + 2             # 0=PAD, 1..n_items=items, n_items+1=MASK
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(self.n_vocab, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.trm_encoder = TransformerEncoderbyHand(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout,
            hidden_act='gelu',
            layer_norm_eps=layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        # 预测头：hidden → vocab（只预测物品，不预测 PAD 和 MASK）
        self.out_proj = nn.Linear(hidden_size, n_items + 1)  # 1..n_items

        # 初始化
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                m.weight = jt.array(
                    np.random.normal(0.0, initializer_range, m.weight.shape).astype(np.float32))
            if isinstance(m, nn.LayerNorm):
                m.weight = jt.ones(m.weight.shape)
                m.bias = jt.zeros(m.bias.shape)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias = jt.zeros(m.bias.shape)

    def get_attn_mask(self, item_seq):
        # 双向注意力（不是 causal），只屏蔽 padding
        pad_mask = (item_seq != 0)  # [B, L]
        ext_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        return jt.where(ext_mask, jt.float32(0.0), jt.float32(-10000.0))

    def execute(self, item_seq):
        """Jittor 用 execute 而非 forward"""
        B, L = item_seq.shape
        position_ids = jt.arange(L, dtype=jt.int32).unsqueeze(0).expand(B, -1)
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)
        x = self.LayerNorm(item_emb + pos_emb)
        x = self.dropout(x)
        mask = self.get_attn_mask(item_seq)
        out = self.trm_encoder(x, mask, output_all_encoded_layers=True)
        return out[-1]  # [B, L, H]

    def forward(self, item_seq):
        return self.execute(item_seq)

    def get_user_repr(self, item_seq):
        """推理时：在序列末尾附加 [MASK]，取 mask 位置的输出作为用户表示"""
        B, L = item_seq.shape
        # 找每个序列中最后一个非 0 的位置
        seq_len = (item_seq != 0).sum(dim=1)  # [B]

        # 在末尾添加 [MASK] token（截断到 max_seq_len）
        mask_col = jt.full((B, 1), self.mask_token, dtype=jt.int32)
        # 如果序列已满，把最后一个位置换成 [MASK]
        item_seq_with_mask = jt.cat([item_seq[:, 1:], mask_col], dim=1)

        out = self.forward(item_seq_with_mask)  # [B, L, H]
        # 取最后一个位置（即 [MASK] 位置）
        user_repr = out[:, -1, :]  # [B, H]
        return user_repr


def log(msg):
    print(msg, flush=True)


# ============================================================
# 用户历史构建
# ============================================================
def build_user_history(src_arr, dst_arr, t_arr):
    raw = defaultdict(list)
    for s, d, ts in zip(src_arr, dst_arr, t_arr):
        raw[int(s)].append((float(ts), int(d)))
    result = {}
    for uid, lst in raw.items():
        lst.sort()
        ts_arr = np.array([x[0] for x in lst], dtype=np.float32)
        it_arr = np.array([x[1] for x in lst], dtype=np.int32)
        result[uid] = (ts_arr, it_arr)
    return result


def get_batch_seqs(src, t, seq_len, user_history):
    B = len(src)
    item_seq = np.zeros((B, seq_len), dtype=np.int32)
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
    return item_seq


# ============================================================
# MLM 掩码函数
# ============================================================
def apply_mlm_mask(item_seq, mask_token, mask_prob, rng):
    """
    随机 mask item_seq 中非 0 位置的 mask_prob 比例
    返回：masked_seq, labels（只在 mask 位置有值，其他=0）
    """
    B, L = item_seq.shape
    masked_seq = item_seq.copy()
    labels = np.zeros_like(item_seq)
    for i in range(B):
        for j in range(L):
            if item_seq[i, j] == 0:
                continue
            if rng.random() < mask_prob:
                labels[i, j] = item_seq[i, j]
                masked_seq[i, j] = mask_token
    return masked_seq, labels


# ============================================================
# 验证
# ============================================================
def test_val(model, val_data, user_history, seq_len, batch_size=256):
    model.eval()
    mrr_sum, mrr_count = 0.0, 0
    val_src, val_dst, val_t, val_negs = val_data

    with jt.no_grad():
        for i in range(0, len(val_src), batch_size):
            src = val_src[i:i+batch_size]
            dst = val_dst[i:i+batch_size]
            t = val_t[i:i+batch_size]
            negs = val_negs[i:i+batch_size]  # [b, 99]
            b = len(src)

            item_seq = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = model.get_user_repr(jt.Var(item_seq).int32())  # [b, H]

            # 正样本分
            pos_emb = model.item_embedding(jt.Var(dst).int32())
            pos_score = jt.sigmoid((user_repr * pos_emb).sum(-1)).numpy()

            # 负样本分
            neg_flat = negs.reshape(-1)
            user_repr_rep = user_repr.unsqueeze(1).expand(-1, 99, -1).reshape(b*99, -1)
            neg_emb = model.item_embedding(jt.Var(neg_flat).int32())
            neg_score = jt.sigmoid((user_repr_rep * neg_emb).sum(-1)).numpy().reshape(b, 99)

            ranks = 1 + np.sum(neg_score > pos_score.reshape(b, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            mrr_count += b
            jt.sync_all()

    return mrr_sum / max(mrr_count, 1)


# ============================================================
# 测试打分
# ============================================================
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

            item_seq = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = model.get_user_repr(jt.Var(item_seq).int32())  # [b, H]

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


# ============================================================
# Main
# ============================================================
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
parser.add_argument("--mask_prob", type=float, default=0.2,
                    help="MLM 掩码概率")
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
log(f"Dataset: {args.dataset}, seq_len={args.seq_len}, hidden={args.hidden_size}")

# ============================================================
# Load data
# ============================================================
data_path = osp.join(args.data_dir, args.dataset)
df = pd.read_csv(osp.join(data_path, "train.csv"))
test_df = pd.read_csv(osp.join(data_path, "test.csv"))

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

# 用全量训练数据构建用户历史
user_history = build_user_history(src_all, dst_all, t_all)

max_item_id = max(int(dst_all.max()), int(test_cands.max()))
log(f"Max item ID: {max_item_id}")

# ============================================================
# 构建模型
# ============================================================
model = BERT4Rec(
    n_items=max_item_id,
    max_seq_len=args.seq_len,
    hidden_size=args.hidden_size,
    inner_size=args.hidden_size * 2,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    dropout=args.dropout,
)
log(f"BERT4Rec params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 验证集准备
# ============================================================
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

# ============================================================
# 训练
# ============================================================
optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
rng_train = np.random.RandomState(0)
MASK_TOKEN = model.mask_token

# 构建训练序列（每个用户的完整历史切片）
log("Building training sequences...")
train_seqs = []
train_src_arr = train_df["src"].values.astype(np.int32)
train_dst_arr = train_df["dst"].values.astype(np.int32)
train_t_arr = train_df["time"].values.astype(np.float32)

# 每条训练边：用该时间之前的历史作为序列，该时间的 dst 作为预测目标
# 为了训练效率，我们随机采样（不是所有边都用）
MAX_TRAIN_SAMPLES = 500000
n_train = len(train_df)
if n_train > MAX_TRAIN_SAMPLES:
    samp_idx = rng_train.choice(n_train, MAX_TRAIN_SAMPLES, replace=False)
    samp_idx.sort()
else:
    samp_idx = np.arange(n_train)

train_src_samp = train_src_arr[samp_idx]
train_dst_samp = train_dst_arr[samp_idx]
train_t_samp = train_t_arr[samp_idx]

best_mrr = 0.0

for epoch in range(args.epochs):
    model.train()
    losses = []
    n_samp = len(train_src_samp)

    # shuffle
    perm = rng_train.permutation(n_samp)
    train_src_ep = train_src_samp[perm]
    train_dst_ep = train_dst_samp[perm]
    train_t_ep = train_t_samp[perm]

    pbar = tqdm(range(0, n_samp, args.batch_size), ncols=120, desc=f"Epoch {epoch+1}")
    for step, start in enumerate(pbar):
        end = min(start + args.batch_size, n_samp)
        src_b = train_src_ep[start:end]
        dst_b = train_dst_ep[start:end]
        t_b = train_t_ep[start:end]
        b = len(src_b)

        # 构建序列
        item_seq = get_batch_seqs(src_b, t_b, args.seq_len, user_history)  # [b, L]

        # 将真实下一物品附加到序列末尾（作为 BERT4Rec 的训练目标之一）
        # 策略：在序列中随机 mask，包括末尾的目标位置
        # 把 dst 加到序列末尾
        next_item = dst_b.reshape(-1, 1).astype(np.int32)
        seq_with_target = np.concatenate([item_seq[:, 1:], next_item], axis=1)  # [b, L]

        # MLM masking
        masked_seq, labels = apply_mlm_mask(seq_with_target, MASK_TOKEN, args.mask_prob, rng_train)

        # 确保至少 mask 最后一个位置（目标位置）
        for i in range(b):
            if labels[i, -1] == 0:  # 如果目标位置没被 mask
                labels[i, -1] = seq_with_target[i, -1]
                masked_seq[i, -1] = MASK_TOKEN

        # Forward
        hidden = model(jt.Var(masked_seq).int32())  # [b, L, H]

        # 只在 mask 位置计算损失
        mask_positions = labels != 0  # [b, L]
        if not mask_positions.any():
            continue

        # 取 mask 位置的 hidden states
        hidden_flat = hidden.reshape(b * args.seq_len, -1)  # [b*L, H]
        labels_flat = labels.reshape(-1)  # [b*L]
        mask_flat = labels_flat != 0  # [b*L]

        mask_indices = np.where(mask_flat)[0]
        if len(mask_indices) == 0:
            continue

        pred_hidden = hidden_flat[jt.Var(mask_indices.astype(np.int32))]  # [M, H]
        target_ids = jt.Var(labels_flat[mask_indices].astype(np.int32))  # [M]

        # 计算 CE loss（全词表 softmax）
        logits = model.out_proj(pred_hidden)  # [M, n_items+1]
        loss = jt.nn.cross_entropy_loss(logits, target_ids)

        optimizer.zero_grad()
        optimizer.step(loss)
        jt.sync_all()

        losses.append(float(loss.item()))
        pbar.set_description(f"Epoch {epoch+1} loss={np.mean(losses[-50:]):.4f}")

        if (step + 1) % 32 == 0:
            jt.gc()

    log(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")

    if val_data is not None:
        mrr = test_val(model, val_data, user_history, args.seq_len)
        log(f"Epoch {epoch+1} Val MRR (100-way): {mrr:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_bert4rec_best.pkl"))
            log(f"  -> New best: {best_mrr:.4f}")
    else:
        jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_bert4rec_ep{epoch+1}.pkl"))

    jt.gc()

# ============================================================
# 生成预测
# ============================================================
log("\nGenerating test predictions...")
best_ckpt = osp.join(save_dir, f"{args.dataset}_bert4rec_best.pkl")
last_ckpt = osp.join(save_dir, f"{args.dataset}_bert4rec_ep{args.epochs}.pkl")
if osp.exists(best_ckpt):
    model.load_state_dict(jt.load(best_ckpt))
    log("Loaded best checkpoint.")
elif osp.exists(last_ckpt):
    model.load_state_dict(jt.load(last_ckpt))
    log("Loaded last checkpoint.")

scores = test_competition(model, test_src, test_time, test_cands, user_history, args.seq_len)

log(f"Scores shape: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_bert4rec.csv")
with open(out_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {out_file}")
log("DONE")
