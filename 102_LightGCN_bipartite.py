"""
102_LightGCN_bipartite.py — LightGCN 二部图协同过滤

算法原理：
  - LightGCN 是最轻量的图神经网络推荐模型：没有特征变换，只做邻居聚合
  - 用户 u 的表示 = 他买过的所有商品表示的平均（多层传播后取均值）
  - 商品 i 的表示 = 买过它的所有用户表示的平均
  - 训练用 BPR 损失（用户-正样本分高于用户-负样本分）
  - 本质是高效的矩阵分解 + 图平滑

适合 Dataset2 的理由：
  - 严格二部图（用户→商品），LightGCN 专为二部图设计
  - 用户历史丰富（中位 1037），图传播信号强
  - 对已知商品效果好（训练出了商品表示）

已知局限：
  - 冷启动商品无图表示（训练图中不存在）
  - 纯协同过滤，不利用序列顺序信息

输出目录：102_LightGCN_bipartite_MMDD
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

try:
    from jittor_geometric.nn.models.lightgcn import LightGCN
    HAS_LIGHTGCN = True
except ImportError:
    HAS_LIGHTGCN = False

# ============================================================
# 若 LightGCN 不可用，自己实现一个简单版
# ============================================================
class SimpleLightGCN(jt.nn.Module):
    """
    简单 LightGCN：纯邻居聚合，无参数变换
    用户 ID: 0..n_users-1
    商品 ID: n_users..n_users+n_items-1（偏移后）
    """
    def __init__(self, n_users, n_items, emb_dim, n_layers=3, reg_weight=1e-4,
                 initializer_range=0.02):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.reg_weight = reg_weight

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # 初始化
        for emb in [self.user_emb, self.item_emb]:
            emb.weight = jt.array(
                np.random.normal(0.0, initializer_range, emb.weight.shape).astype(np.float32))

        self._adj_built = False
        self.D_u_inv = None  # user degree inverse
        self.D_i_inv = None  # item degree inverse

    def build_adj(self, user_ids_np, item_ids_np):
        """预计算归一化邻接矩阵（稀疏表示）"""
        # 度
        user_degree = np.bincount(user_ids_np, minlength=self.n_users).astype(np.float32)
        item_degree = np.bincount(item_ids_np, minlength=self.n_items).astype(np.float32)
        user_degree[user_degree == 0] = 1
        item_degree[item_degree == 0] = 1

        self.D_u_inv_sqrt = jt.array(1.0 / np.sqrt(user_degree))  # [n_users]
        self.D_i_inv_sqrt = jt.array(1.0 / np.sqrt(item_degree))  # [n_items]

        # 边权重：1/sqrt(deg_u) * 1/sqrt(deg_i)
        edge_weight = (1.0 / np.sqrt(user_degree[user_ids_np]) *
                       1.0 / np.sqrt(item_degree[item_ids_np]))
        self.edge_u = jt.array(user_ids_np.astype(np.int32))
        self.edge_i = jt.array(item_ids_np.astype(np.int32))
        self.edge_w = jt.array(edge_weight.astype(np.float32))
        self._adj_built = True
        print(f"  Adj built: {len(user_ids_np)} edges, {self.n_users} users, {self.n_items} items")

    def graph_propagate(self):
        """运行 L 层图传播，返回最终用户/商品嵌入（各层均值）"""
        D = self.user_emb.weight.shape[-1]
        u_emb = self.user_emb.weight  # [n_users, D]
        i_emb = self.item_emb.weight  # [n_items, D]

        u_layers = [u_emb]
        i_layers = [i_emb]

        E = self.edge_u.shape[0]
        idx_u = self.edge_u.unsqueeze(-1).broadcast([E, D])  # [E, D] - user indices
        idx_i = self.edge_i.unsqueeze(-1).broadcast([E, D])  # [E, D] - item indices

        for _ in range(self.n_layers):
            # 商品 → 用户聚合
            weighted_i = i_emb[self.edge_i] * self.edge_w.unsqueeze(-1)  # [E, D]
            u_new = jt.zeros((self.n_users, D))
            u_new = jt.scatter(u_new, 0, idx_u, weighted_i, reduce='add')

            # 用户 → 商品聚合
            weighted_u = u_emb[self.edge_u] * self.edge_w.unsqueeze(-1)  # [E, D]
            i_new = jt.zeros((self.n_items, D))
            i_new = jt.scatter(i_new, 0, idx_i, weighted_u, reduce='add')

            u_emb = u_new
            i_emb = i_new
            u_layers.append(u_emb)
            i_layers.append(i_emb)

        # 所有层取均值
        u_final = sum(u_layers) / len(u_layers)
        i_final = sum(i_layers) / len(i_layers)
        return u_final, i_final

    def forward_bpr(self, user_ids, pos_ids, neg_ids):
        u_all, i_all = self.graph_propagate()
        u_emb = u_all[user_ids]
        pos_emb = i_all[pos_ids]
        neg_emb = i_all[neg_ids]

        pos_score = (u_emb * pos_emb).sum(-1)
        neg_score = (u_emb * neg_emb).sum(-1)
        bpr_loss = -jt.log(jt.sigmoid(pos_score - neg_score) + 1e-8).mean()

        # L2 regularization on original embeddings
        u_orig = self.user_emb(user_ids)
        p_orig = self.item_emb(pos_ids)
        n_orig = self.item_emb(neg_ids)
        reg = self.reg_weight * (
            (u_orig ** 2).sum() + (p_orig ** 2).sum() + (n_orig ** 2).sum()
        ) / len(user_ids)

        return bpr_loss + reg

    def get_all_embeddings(self):
        return self.graph_propagate()


def log(msg):
    print(msg, flush=True)


# ============================================================
# Main
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--reg_weight", type=float, default=1e-4)
parser.add_argument("--num_neg", type=int, default=1,
                    help="BPR 每个正样本对应的负样本数")
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
log(f"Dataset: {args.dataset}, emb_dim={args.emb_dim}, n_layers={args.n_layers}")

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

# ============================================================
# ID 重映射（用户、商品 各自从 0 开始）
# ============================================================
unique_users = np.unique(np.concatenate([src_all, test_src]))
unique_items_train = np.unique(dst_all)
all_cand_items = np.unique(test_cands.reshape(-1))

# 包含所有训练+测试候选 item
unique_items = np.unique(np.concatenate([unique_items_train, all_cand_items]))

user2idx = {int(u): i for i, u in enumerate(unique_users)}
item2idx = {int(v): i for i, v in enumerate(unique_items)}

n_users = len(unique_users)
n_items = len(unique_items)
log(f"Users: {n_users}, Items: {n_items} (train: {len(unique_items_train)}, cold: {n_items - len(unique_items_train)})")

# 用户/商品 ID 转为 0-indexed
def remap(arr, id2idx):
    return np.array([id2idx.get(int(x), 0) for x in arr.reshape(-1)], dtype=np.int32).reshape(arr.shape)

train_u = remap(train_df["src"].values, user2idx)
train_v = remap(train_df["dst"].values, item2idx)

# 构建模型
model = SimpleLightGCN(
    n_users=n_users, n_items=n_items, emb_dim=args.emb_dim,
    n_layers=args.n_layers, reg_weight=args.reg_weight
)
model.build_adj(train_u, train_v)
log(f"LightGCN params: {sum(p.numel() for p in model.parameters()):,}")

# 负样本准备
item_count = Counter(train_v.tolist())
pop_items = np.array(list(item_count.keys()), dtype=np.int32)
pop_probs = np.sqrt(np.array(list(item_count.values()), dtype=np.float64))
pop_probs /= pop_probs.sum()
rng_train = np.random.RandomState(0)

def sample_negs_bpr(pos_v, K):
    B = len(pos_v)
    raw = rng_train.choice(pop_items, size=B * K * 2, p=pop_probs).reshape(B, K * 2)
    result = np.zeros((B, K), dtype=np.int32)
    for i in range(B):
        cands = raw[i][raw[i] != pos_v[i]][:K]
        if len(cands) < K:
            cands = np.concatenate([cands, rng_train.choice(pop_items, K - len(cands))])
        result[i] = cands[:K]
    return result.reshape(-1)

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)
best_mrr = 0.0
n_train = len(train_u)

for epoch in range(args.epochs):
    model.train()
    losses = []
    perm = rng_train.permutation(n_train)
    ep_u = train_u[perm]
    ep_v = train_v[perm]

    pbar = tqdm(range(0, n_train, args.batch_size), ncols=120, desc=f"Epoch {epoch+1}")
    for step, start in enumerate(pbar):
        end = min(start + args.batch_size, n_train)
        u_b = ep_u[start:end]
        v_b = ep_v[start:end]
        b = len(u_b)

        neg_b = sample_negs_bpr(v_b, args.num_neg)
        # 展开 u_b 匹配 neg (如果 num_neg>1)
        if args.num_neg > 1:
            u_b_rep = np.repeat(u_b, args.num_neg)
            v_b_rep = np.repeat(v_b, args.num_neg)
        else:
            u_b_rep = u_b
            v_b_rep = v_b

        loss = model.forward_bpr(
            jt.Var(u_b_rep).int32(),
            jt.Var(v_b_rep).int32(),
            jt.Var(neg_b).int32(),
        )
        optimizer.zero_grad()
        optimizer.step(loss)
        jt.sync_all()

        losses.append(float(loss.item()))
        pbar.set_description(f"Epoch {epoch+1} loss={np.mean(losses[-50:]):.4f}")
        if (step + 1) % 32 == 0:
            jt.gc()

    log(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")

    # 本地验证
    if val_df is not None:
        model.eval()
        with jt.no_grad():
            u_all, i_all = model.get_all_embeddings()
        rng_val = np.random.RandomState(42)
        all_items_idx = np.arange(len(unique_items_train), dtype=np.int32)

        VAL_N = min(5000, len(val_df))
        val_idx = rng_val.choice(len(val_df), VAL_N, replace=False)
        val_src = val_df["src"].values[val_idx]
        val_dst = val_df["dst"].values[val_idx]
        val_neg = rng_val.choice(unique_items_train, size=(VAL_N, 99), replace=True)

        # 重映射
        val_u_idx = remap(val_src, user2idx)
        val_pos_idx = remap(val_dst, item2idx)
        val_neg_idx = remap(val_neg.reshape(-1), item2idx).reshape(VAL_N, 99)

        u_emb = u_all.numpy()[val_u_idx]
        pos_emb = i_all.numpy()[val_pos_idx]
        neg_embs = i_all.numpy()[val_neg_idx.reshape(-1)].reshape(VAL_N, 99, -1)

        pos_score = np.sum(u_emb * pos_emb, axis=-1)  # [VAL_N]
        neg_score = np.sum(u_emb[:, np.newaxis, :] * neg_embs, axis=-1)  # [VAL_N, 99]
        ranks = 1 + np.sum(neg_score > pos_score.reshape(-1, 1), axis=1)
        mrr = float(np.mean(1.0 / ranks))
        log(f"Epoch {epoch+1} Val MRR: {mrr:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_lgcn_best.pkl"))
            log(f"  -> New best: {best_mrr:.4f}")
    else:
        jt.save(model.state_dict(), osp.join(save_dir, f"{args.dataset}_lgcn_ep{epoch+1}.pkl"))
    jt.gc()

log("\nGenerating test predictions...")
best_ckpt = osp.join(save_dir, f"{args.dataset}_lgcn_best.pkl")
last_ckpt = osp.join(save_dir, f"{args.dataset}_lgcn_ep{args.epochs}.pkl")
if osp.exists(best_ckpt):
    model.load_state_dict(jt.load(best_ckpt))
elif osp.exists(last_ckpt):
    model.load_state_dict(jt.load(last_ckpt))

model.eval()
with jt.no_grad():
    u_all, i_all = model.get_all_embeddings()
u_np = u_all.numpy()
i_np = i_all.numpy()

N_test = len(test_src)
N_cands = test_cands.shape[1]
all_scores = np.zeros((N_test, N_cands), dtype=np.float32)

for i in tqdm(range(N_test), ncols=100, desc="Testing"):
    u_idx = user2idx.get(int(test_src[i]), 0)
    u_vec = u_np[u_idx]
    cands = test_cands[i]
    c_idx = np.array([item2idx.get(int(c), 0) for c in cands], dtype=np.int32)
    c_vecs = i_np[c_idx]  # [100, D]
    scores = np.dot(c_vecs, u_vec)  # [100]
    all_scores[i] = scores

# 归一化
scores_min = all_scores.min(axis=1, keepdims=True)
scores_max = all_scores.max(axis=1, keepdims=True)
all_scores = (all_scores - scores_min) / (scores_max - scores_min + 1e-8)
all_scores = np.clip(all_scores, 0, 1)

log(f"Scores shape: {all_scores.shape}, range=[{all_scores.min():.4f}, {all_scores.max():.4f}]")
out_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_lgcn.csv")
with open(out_file, "w") as f:
    for row in all_scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {out_file}")
log("DONE")
