"""
108_SASRec_IDhash.py — SASRec + ID Hash Decomposition for Cold-Start Items

算法原理：
  - 普通 SASRec：每个物品 ID 对应一个独立 embedding（冷启动物品 = 随机初始化）
  - ID 哈希分解：把物品 ID 分解成多个层级，每层有共享的 embedding 表
    例如：ID=85432 → coarse(85432//500=170) + fine(85432%500=432)
    cold start 物品的某些哈希桶可能在训练集中出现过，从而借用其语义信息
  - Dataset2 特有问题：54% 测试候选是冷启动（训练未见过的 ID）
    ID 哈希让冷启动物品通过共享桶获得有意义的表示，而非纯随机

哈希方案（3级分解，总参数量与普通 SASRec 相近）：
  level1（粗粒度）：ID // 500 → 约 221 个桶
  level2（中粒度）：(ID // 50) % 100 → 100 个桶
  level3（细粒度）：ID % 50 → 50 个桶
  最终：embed = embed1 + embed2 + embed3

输出目录：108_SASRec_IDhash_MMDD
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
from jittor_geometric.nn.models.sasrec import SASRec
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader
from jittor_geometric.data import TemporalData
import argparse

jt.flags.use_cuda = 1

# ============================================================
# Hash Embedding Module
# ============================================================
class HashItemEmbedding(jt.nn.Module):
    """
    三级哈希嵌入：item embedding = embed1(id//L1) + embed2((id//L2)%M2) + embed3(id%L3)
    允许冷启动物品通过共享桶借用训练物品的语义信息
    """
    def __init__(self, max_id, hidden_size, L1=500, L2=50, M2=100, L3=50,
                 padding_idx=0, initializer_range=0.02):
        super().__init__()
        self.L1, self.L2, self.M2, self.L3 = L1, L2, M2, L3
        n1 = max_id // L1 + 2
        n2 = M2 + 1
        n3 = L3 + 1
        self.embed1 = nn.Embedding(n1, hidden_size, padding_idx=0)
        self.embed2 = nn.Embedding(n2, hidden_size, padding_idx=0)
        self.embed3 = nn.Embedding(n3, hidden_size, padding_idx=0)
        # 标量权重（学习各级的贡献）
        self.w1 = jt.array(np.array([1.0], dtype=np.float32))
        self.w2 = jt.array(np.array([0.5], dtype=np.float32))
        self.w3 = jt.array(np.array([0.3], dtype=np.float32))
        self.hidden_size = hidden_size
        # 初始化
        for emb in [self.embed1, self.embed2, self.embed3]:
            emb.weight = jt.array(
                np.random.normal(0.0, initializer_range, emb.weight.shape).astype(np.float32))

    def execute(self, item_ids):
        """item_ids: int32 tensor，0=padding"""
        # 全程用 Jittor 整数运算，保持在 GPU 上
        ids = item_ids.int32()
        b1 = ids // self.L1          # level1: coarse
        b2 = (ids // self.L2) % self.M2  # level2: medium
        b3 = ids % self.L3           # level3: fine

        e1 = self.embed1(b1)
        e2 = self.embed2(b2)
        e3 = self.embed3(b3)
        return e1 + e2 * 0.5 + e3 * 0.3


# ============================================================
# SASRec + Hash Embedding
# ============================================================
class SASRecHashEmb(jt.nn.Module):
    """用 HashItemEmbedding 替换 SASRec 原来的 nn.Embedding"""
    def __init__(self, max_item_id, max_seq_len, hidden_size=128, inner_size=256,
                 n_layers=2, n_heads=2, dropout=0.1, layer_norm_eps=1e-12,
                 initializer_range=0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.item_embedding = HashItemEmbedding(
            max_id=max_item_id, hidden_size=hidden_size,
            initializer_range=initializer_range
        )
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

        # 初始化 position embedding
        self.position_embedding.weight = jt.array(
            np.random.normal(0.0, initializer_range,
                             self.position_embedding.weight.shape).astype(np.float32))

    def get_attention_mask(self, item_seq):
        # 单向（causal），与 SASRec 一致
        non_pad = item_seq != 0
        ext_mask = non_pad.unsqueeze(1).unsqueeze(2)
        ext_mask = jt.tril(ext_mask.expand(-1, -1, item_seq.shape[1], -1))
        return jt.where(ext_mask, jt.float32(0.0), jt.float32(-10000.0))

    def forward(self, item_seq, item_seq_len):
        B, L = item_seq.shape
        pos_ids = jt.arange(L, dtype=jt.int32).unsqueeze(0).expand(B, -1)
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(pos_ids)
        x = self.LayerNorm(item_emb + pos_emb)
        x = self.dropout(x)
        mask = self.get_attention_mask(item_seq)
        out = self.trm_encoder(x, mask, output_all_encoded_layers=True)
        seq_out = out[-1]  # [B, L, H]
        # 取最后有效位置
        item_seq_len_clamped = jt.clamp(item_seq_len - 1, min_v=0)
        gather_idx = item_seq_len_clamped.reshape(-1, 1, 1).expand(-1, -1, seq_out.shape[-1])
        return seq_out.gather(dim=1, index=gather_idx).squeeze(1)  # [B, H]


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
        ts_arr = np.array([x[0] for x in lst], dtype=np.float32)
        it_arr = np.array([x[1] for x in lst], dtype=np.int32)
        result[uid] = (ts_arr, it_arr)
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


# ============================================================
# InfoNCE loss
# ============================================================
def infonce_loss(pos_logit, neg_logit, K):
    B = pos_logit.shape[0]
    neg = neg_logit.reshape(B, K)
    pos = pos_logit.unsqueeze(1)
    logits = jt.concat([pos, neg], dim=1)
    labels = jt.zeros(B, dtype='int32')
    return jt.nn.cross_entropy_loss(logits, labels)


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
            negs = val_negs[i:i+batch_size]
            b = len(src)

            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = model.forward(
                jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())

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


# ============================================================
# Test scoring
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
            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = model.forward(
                jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())
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
parser.add_argument("--num_neg", type=int, default=31)
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
log(f"Dataset: {args.dataset}, hash decomposition, seq_len={args.seq_len}")

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

model = SASRecHashEmb(
    max_item_id=max_item_id,
    max_seq_len=args.seq_len,
    hidden_size=args.hidden_size,
    inner_size=args.hidden_size * 2,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    dropout=args.dropout,
)
log(f"SASRec-HashEmb params: {sum(p.numel() for p in model.parameters()):,}")

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

# 负样本采样（流行度加权）
from collections import Counter
dst_counter = Counter(train_df["dst"].values.tolist())
pop_items = np.array(list(dst_counter.keys()), dtype=np.int32)
pop_probs = np.sqrt(np.array(list(dst_counter.values()), dtype=np.float64))
pop_probs /= pop_probs.sum()
rng_train = np.random.RandomState(0)

def sample_negs(src, dst, K):
    B = len(src)
    negs = rng_train.choice(pop_items, size=B * K * 3, p=pop_probs).reshape(B, K * 3)
    dst_arr = dst.reshape(-1, 1)
    result = np.zeros((B, K), dtype=np.int32)
    for i in range(B):
        cands = negs[i][negs[i] != dst_arr[i, 0]][:K]
        if len(cands) < K:
            cands = np.concatenate([cands,
                rng_train.choice(pop_items, size=K-len(cands))])
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

        neg_b = sample_negs(src_b, dst_b, args.num_neg)

        item_seq, item_seq_len = get_batch_seqs(src_b, t_b, args.seq_len, user_history)
        user_repr = model.forward(
            jt.Var(item_seq).int32(), jt.Var(item_seq_len).int32())

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

    if val_data is not None:
        mrr = test_val(model, val_data, user_history, args.seq_len)
        log(f"Epoch {epoch+1} Val MRR: {mrr:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_idhash_best.pkl"))
            log(f"  -> New best: {best_mrr:.4f}")
    else:
        jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_idhash_ep{epoch+1}.pkl"))
    jt.gc()

log("\nGenerating test predictions...")
best_ckpt = osp.join(save_dir, f"{args.dataset}_idhash_best.pkl")
last_ckpt = osp.join(save_dir, f"{args.dataset}_idhash_ep{args.epochs}.pkl")
if osp.exists(best_ckpt):
    model.load_state_dict(jt.load(best_ckpt))
    log("Loaded best checkpoint.")
elif osp.exists(last_ckpt):
    model.load_state_dict(jt.load(last_ckpt))

scores = test_competition(model, test_src, test_time, test_cands, user_history, args.seq_len)
log(f"Scores shape: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_idhash.csv")
with open(out_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {out_file}")
log("DONE")
