"""
26TGN_ds1.py — Temporal Graph Network for Dataset1 (social graph)

Architecture:
- TGNMemory: accumulates per-node interaction history (perfect for 72.6% repeat edges)
- GraphAttentionEmbedding: TransformerConv over temporal neighbors
- LinkPredictor: MLP(z_src, z_dst, explicit_features)
- Explicit features: freq(u,v), time_decay(u,v), reciprocal_freq(v,u)
- Loss: InfoNCE with K=15 negatives from test candidate pool

Usage:
  CUDA_VISIBLE_DEVICES=2 python 26TGN_ds1.py --dataset dataset1 \
      --data_dir ./data --save_dir ./saved_models/tgn_ds1 --epochs 20
"""
import os
import os.path as osp
import sys
import time as time_module

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
from jittor.nn import Linear
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.nn import TGNMemory, TransformerConv
from jittor_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader
import argparse

jt.flags.use_cuda = 1


def log(msg):
    print(msg, flush=True)


# ============================================================
# Explicit interaction features (frequency + time decay)
# ============================================================
class ExplicitFeatures:
    """Precompute frequency + time decay features from training edges."""

    def __init__(self, src_arr, dst_arr, t_arr, decay_lambda=1e-7):
        self.decay_lambda = decay_lambda
        # freq_dict[(u,v)] = count
        freq = Counter(zip(src_arr.tolist(), dst_arr.tolist()))
        # last_t_dict[(u,v)] = last interaction time
        last_t = {}
        for s, d, ts in zip(src_arr, dst_arr, t_arr):
            key = (int(s), int(d))
            if key not in last_t or float(ts) > last_t[key]:
                last_t[key] = float(ts)
        # total degree
        total_out = Counter(src_arr.tolist())
        total_in = Counter(dst_arr.tolist())
        self.freq = dict(freq)
        self.last_t = last_t
        self.total_out = dict(total_out)
        self.total_in = dict(total_in)
        self.max_t = float(t_arr.max())

    def get(self, src_arr, dst_arr, query_t_arr):
        """Return [N, 4] float32 feature matrix."""
        N = len(src_arr)
        feats = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            u, v, qt = int(src_arr[i]), int(dst_arr[i]), float(query_t_arr[i])
            f_uv = self.freq.get((u, v), 0)
            f_vu = self.freq.get((v, u), 0)
            t_out_u = max(1, self.total_out.get(u, 1))
            t_in_u = max(1, self.total_in.get(u, 1))
            lt = self.last_t.get((u, v), -1e18)
            dt = max(0.0, qt - lt) if lt > -1e17 else 1e18
            feats[i, 0] = f_uv / t_out_u                                   # freq ratio u→v
            feats[i, 1] = f_uv * np.exp(-self.decay_lambda * dt) / t_out_u  # time-decayed freq
            feats[i, 2] = f_vu / max(1, self.total_out.get(v, 1))           # reciprocal freq
            feats[i, 3] = 1.0 / (1.0 + dt / (self.max_t + 1.0))            # recency score
        return feats

    def get_batch(self, src_arr, cand_arr, query_t_scalar):
        """For a single user and multiple candidates at query_t. Returns [K, 4]."""
        t_arr = np.full(len(cand_arr), query_t_scalar, dtype=np.float32)
        return self.get(
            np.full(len(cand_arr), src_arr, dtype=np.int32),
            cand_arr.astype(np.int32),
            t_arr
        )


# ============================================================
# Negative samplers
# ============================================================
class PopNegSampler:
    def __init__(self, dst_arr, p_pop=0.7, seed=42):
        self._rng = np.random.RandomState(seed)
        self.p_pop = p_pop
        dst_counts = Counter(dst_arr.tolist())
        dsts = np.array(list(dst_counts.keys()), dtype=np.int32)
        freqs = np.array([dst_counts[d] for d in dsts], dtype=np.float64)
        sqrt_freqs = np.sqrt(freqs)
        self.pop_dsts = dsts
        self.pop_probs = sqrt_freqs / sqrt_freqs.sum()
        self.min_dst = int(dst_arr.min())
        self.max_dst = int(dst_arr.max())

    def sample(self, B, K):
        n_pop = max(1, int(K * self.p_pop))
        samps = self._rng.choice(self.pop_dsts, size=B * K * 3,
                                 p=self.pop_probs, replace=True)
        result = samps[:B * K].astype(np.int32)
        return result.reshape(B, K)


class TestPoolNegSampler:
    def __init__(self, test_candidates, seed=42):
        self._rng = np.random.RandomState(seed)
        flat = test_candidates.reshape(-1).astype(np.int64)
        vals, cnts = np.unique(flat, return_counts=True)
        self.pool = vals.astype(np.int32)
        w = np.sqrt(cnts.astype(np.float64))
        self.pool_probs = w / w.sum()

    def sample(self, dst, K):
        B = len(dst)
        cand = self._rng.choice(self.pool, size=B * K,
                                p=self.pool_probs, replace=True).astype(np.int32)
        cand = cand.reshape(B, K)
        dst_arr = np.asarray(dst, dtype=np.int32).reshape(B, 1)
        clash = cand == dst_arr
        if clash.any():
            fix = self._rng.choice(self.pool, size=int(clash.sum()),
                                   replace=True).astype(np.int32)
            cand[clash] = fix
        return cand


# ============================================================
# Model components
# ============================================================
class GraphAttentionEmbedding(jt.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                   dropout=0.1, edge_dim=edge_dim)

    def execute(self, x, last_update, edge_index, t, msg):
        if edge_index.shape[1] == 0:
            return x
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.float())
        edge_attr = jt.concat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(jt.nn.Module):
    def __init__(self, emb_dim, feat_dim=4, hidden=64):
        super().__init__()
        in_dim = emb_dim * 2 + feat_dim
        self.mlp = nn.Sequential(
            Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden, 1),
        )

    def execute(self, z_src, z_dst, feats):
        h = jt.concat([z_src, z_dst, feats], dim=-1)
        return self.mlp(h).squeeze(-1)


# ============================================================
# InfoNCE loss
# ============================================================
def infonce_loss(pos_score, neg_scores):
    """pos_score: [B], neg_scores: [B, K]"""
    B = pos_score.shape[0]
    logits = jt.concat([pos_score.unsqueeze(1), neg_scores], dim=1)  # [B, K+1]
    labels = jt.zeros(B, dtype='int32')
    return jt.nn.cross_entropy_loss(logits, labels)


# ============================================================
# Validation (100-way MRR)
# ============================================================
def evaluate_mrr(memory, gnn, predictor, assoc, explicit,
                 val_src, val_dst, val_t, neg_sampler, K_val=99):
    """Process val data sequentially to maintain temporal ordering."""
    memory.eval()
    gnn.eval()
    predictor.eval()

    n = len(val_src)
    mrr_sum, count = 0.0, 0
    batch_size = 200

    with jt.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = end - start
            src_b = val_src[start:end]
            dst_b = val_dst[start:end]
            t_b = val_t[start:end]

            # Sample negatives
            neg_b = neg_sampler.sample(dst_b, K_val)  # [b, K_val]

            # Get embeddings for all involved nodes
            all_nodes_np = np.unique(np.concatenate([src_b, dst_b, neg_b.reshape(-1)]))
            all_nodes = jt.Var(all_nodes_np.astype(np.int32))
            z, last_update = memory(all_nodes)

            # Create local index mapping
            node_to_local = {int(n): i for i, n in enumerate(all_nodes_np)}

            src_local = np.array([node_to_local[int(s)] for s in src_b], dtype=np.int32)
            dst_local = np.array([node_to_local[int(d)] for d in dst_b], dtype=np.int32)

            z_src = z[jt.Var(src_local)]   # [b, H]
            z_dst = z[jt.Var(dst_local)]   # [b, H]

            # Positive explicit features
            pos_feats = explicit.get(src_b, dst_b, t_b)  # [b, 4]
            pos_score = predictor(z_src, z_dst, jt.Var(pos_feats))  # [b]

            # Negative scores
            neg_scores = np.zeros((b, K_val), dtype=np.float32)
            for k in range(K_val):
                neg_k = neg_b[:, k]
                neg_local = np.array([node_to_local.get(int(nd), 0) for nd in neg_k],
                                     dtype=np.int32)
                z_neg_k = z[jt.Var(neg_local)]
                neg_feats = explicit.get(src_b, neg_k, t_b)
                neg_score_k = predictor(z_src, z_neg_k, jt.Var(neg_feats))
                neg_scores[:, k] = neg_score_k.numpy()

            pos_np = pos_score.numpy()  # [b]
            ranks = 1 + np.sum(neg_scores > pos_np.reshape(b, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            count += b

            jt.gc()

    return mrr_sum / count if count > 0 else 0.0


# ============================================================
# Test inference (100-way scoring)
# ============================================================
def test_competition(memory, gnn, predictor, assoc, explicit,
                     test_src, test_time, test_candidates,
                     batch_size=100):
    memory.eval()
    gnn.eval()
    predictor.eval()

    n = len(test_src)
    num_cands = test_candidates.shape[1]
    all_scores = np.zeros((n, num_cands), dtype=np.float32)

    with jt.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Test inference", ncols=100):
            end = min(start + batch_size, n)
            b = end - start
            src_b = test_src[start:end]
            t_b = test_time[start:end]
            cands_b = test_candidates[start:end]  # [b, 100]

            all_nodes_np = np.unique(
                np.concatenate([src_b, cands_b.reshape(-1)])
            )
            all_nodes = jt.Var(all_nodes_np.astype(np.int32))
            z, _ = memory(all_nodes)
            node_to_local = {int(nd): i for i, nd in enumerate(all_nodes_np)}

            src_local = np.array([node_to_local[int(s)] for s in src_b], dtype=np.int32)
            z_src = z[jt.Var(src_local)]  # [b, H]

            for ci in range(num_cands):
                cands_ci = cands_b[:, ci]  # [b]
                ci_local = np.array([node_to_local.get(int(c), 0) for c in cands_ci],
                                    dtype=np.int32)
                z_ci = z[jt.Var(ci_local)]
                feats = explicit.get(src_b, cands_ci, t_b)
                scores_ci = jt.sigmoid(predictor(z_src, z_ci, jt.Var(feats))).numpy()
                all_scores[start:end, ci] = scores_ci

            jt.gc()

    return all_scores


# ============================================================
# Main
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset1")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./saved_models/tgn_ds1")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--memory_dim", type=int, default=100)
parser.add_argument("--time_dim", type=int, default=100)
parser.add_argument("--embedding_dim", type=int, default=100)
parser.add_argument("--neighbor_size", type=int, default=10)
parser.add_argument("--num_neg", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--early_stop", type=int, default=5)
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"26TGN_ds1 — Dataset: {args.dataset}")
log(f"  memory_dim={args.memory_dim}, time_dim={args.time_dim}, "
    f"embedding_dim={args.embedding_dim}, num_neg={args.num_neg}")
log("=" * 80)

# ---- Load data ----
df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

log(f"Train edges: {len(df)}, Test queries: {len(test_df)}")

# Sort by time (required for TGN temporal processing)
df = df.sort_values("time").reset_index(drop=True)

# Train/val split (last val_ratio by time)
n_val = int(len(df) * args.val_ratio)
train_df = df.iloc[:-n_val].reset_index(drop=True)
val_df = df.iloc[-n_val:].reset_index(drop=True)
log(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Arrays
train_src = train_df["src"].values.astype(np.int32)
train_dst = train_df["dst"].values.astype(np.int32)
train_t = train_df["time"].values.astype(np.float32)

val_src = val_df["src"].values.astype(np.int32)
val_dst = val_df["dst"].values.astype(np.int32)
val_t = val_df["time"].values.astype(np.float32)

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.float32)
cand_cols = [c for c in test_df.columns if c.startswith("c")]
test_candidates = test_df[cand_cols].values.astype(np.int32)

# Node count (must cover all nodes in train + test candidates)
all_node_ids = np.concatenate([
    df["src"].values, df["dst"].values, test_candidates.reshape(-1)
])
num_nodes = int(all_node_ids.max()) + 1
log(f"num_nodes={num_nodes}")

# ---- Precompute explicit features ----
log("Precomputing explicit interaction features...")
explicit = ExplicitFeatures(train_src, train_dst, train_t)

# ---- Negative sampler ----
neg_sampler = TestPoolNegSampler(test_candidates)

# ---- Build model ----
msg_dim = 1  # no edge features → dummy 1-dim zero message
memory = TGNMemory(
    num_nodes, msg_dim, args.memory_dim, args.time_dim,
    message_module=IdentityMessage(msg_dim, args.memory_dim, args.time_dim),
    aggregator_module=LastAggregator(),
)
gnn = GraphAttentionEmbedding(
    in_channels=args.memory_dim,
    out_channels=args.embedding_dim,
    msg_dim=msg_dim,
    time_enc=memory.time_enc,
)
predictor = LinkPredictor(emb_dim=args.embedding_dim, feat_dim=4, hidden=128)
neighbor_loader = LastNeighborLoader(num_nodes, size=args.neighbor_size)

# Global assoc buffer
assoc = jt.zeros(num_nodes, dtype=jt.int32)

all_params = (list(memory.parameters()) + list(gnn.parameters()) +
              list(predictor.parameters()))
optimizer = jt.nn.Adam(all_params, lr=args.lr)

log(f"TGNMemory params: {sum(p.numel() for p in memory.parameters()):,}")
log(f"GNN params: {sum(p.numel() for p in gnn.parameters()):,}")
log(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

# Prepare edge messages (zeros)
all_train_msg = np.zeros((len(train_df), msg_dim), dtype=np.float32)

if args.model_path:
    ckpt = jt.load(args.model_path)
    memory.load_state_dict(ckpt["memory"])
    gnn.load_state_dict(ckpt["gnn"])
    predictor.load_state_dict(ckpt["predictor"])
    log(f"Loaded model from {args.model_path}")

# ---- Training ----
best_mrr = 0.0
patience_counter = 0

if not args.eval_only:
    for epoch in range(args.epochs):
        # Reset TGN state at the start of each epoch
        memory.reset_state()
        neighbor_loader.reset_state()
        memory.train()
        gnn.train()
        predictor.train()

        epoch_losses = []
        n_train = len(train_src)
        pbar = tqdm(range(0, n_train, args.batch_size), ncols=120,
                    desc=f"Epoch {epoch+1}")

        for bi in pbar:
            be = min(bi + args.batch_size, n_train)
            b = be - bi

            src_b = train_src[bi:be]
            dst_b = train_dst[bi:be]
            t_b = train_t[bi:be]
            msg_b = jt.Var(all_train_msg[bi:be])

            # Sample negatives from test pool
            neg_b = neg_sampler.sample(dst_b, args.num_neg)  # [b, K]

            # Get all involved nodes
            all_np = np.unique(np.concatenate([src_b, dst_b, neg_b.reshape(-1)]))
            n_id = jt.Var(all_np.astype(np.int32))

            # Get neighbor subgraph
            n_id_ext, edge_index, e_id = neighbor_loader(n_id)
            assoc_local = {int(v): i for i, v in enumerate(n_id_ext.numpy())}

            # Get TGN memory-augmented embeddings
            z, last_update = memory(n_id_ext)
            e_id_np = e_id.numpy().astype(np.int64)
            if len(e_id_np) > 0 and edge_index.shape[1] > 0:
                # e_id values are 0-indexed from neighbor_loader.cur_e_id
                # clamp to valid range in case of any edge-case
                e_id_np = np.clip(e_id_np, 0, len(train_t) - 1)
                t_sub = jt.Var(train_t[e_id_np])
                msg_sub = jt.zeros((len(e_id_np), msg_dim))
                z = gnn(z, last_update, edge_index, t_sub, msg_sub)

            # Local indices for src, dst, negatives
            src_local = np.array([assoc_local.get(int(s), 0) for s in src_b], dtype=np.int32)
            dst_local = np.array([assoc_local.get(int(d), 0) for d in dst_b], dtype=np.int32)

            z_src = z[jt.Var(src_local)]   # [b, H]
            z_dst = z[jt.Var(dst_local)]   # [b, H]

            # Positive score
            pos_feats = jt.Var(explicit.get(src_b, dst_b, t_b))
            pos_score = predictor(z_src, z_dst, pos_feats)  # [b]

            # Negative scores
            neg_scores_list = []
            for k in range(args.num_neg):
                neg_k = neg_b[:, k]
                neg_local = np.array([assoc_local.get(int(nd), 0) for nd in neg_k],
                                     dtype=np.int32)
                z_neg_k = z[jt.Var(neg_local)]
                neg_feats = jt.Var(explicit.get(src_b, neg_k, t_b))
                neg_score_k = predictor(z_src, z_neg_k, neg_feats)
                neg_scores_list.append(neg_score_k.unsqueeze(1))

            neg_scores = jt.concat(neg_scores_list, dim=1)  # [b, K]

            loss = infonce_loss(pos_score, neg_scores)

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            memory.detach()

            # Update TGN state after backward
            memory.update_state(
                jt.Var(src_b), jt.Var(dst_b),
                jt.Var(t_b.astype(np.int32)), msg_b
            )
            neighbor_loader.insert(jt.Var(src_b), jt.Var(dst_b))

            epoch_losses.append(float(loss.numpy()))
            pbar.set_description(
                f"Epoch {epoch+1} loss={np.mean(epoch_losses[-50:]):.4f}")

            if (bi // args.batch_size + 1) % 64 == 0:
                jt.gc()

        log(f"Epoch {epoch+1} Train Loss: {np.mean(epoch_losses):.4f}")

        # ---- Validation ----
        # Memory state is now built from all training data (memory.eval() reads without updating)
        val_mrr = evaluate_mrr(memory, gnn, predictor, assoc, explicit,
                               val_src, val_dst, val_t, neg_sampler, K_val=99)
        log(f"Epoch {epoch+1} Val 100-way MRR: {val_mrr:.6f}")

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            patience_counter = 0
            ckpt = {
                "memory": memory.state_dict(),
                "gnn": gnn.state_dict(),
                "predictor": predictor.state_dict(),
            }
            jt.save(ckpt, f"{save_path}/{args.dataset}_TGN_best.pkl")
            log(f"  -> New best MRR: {best_mrr:.6f}, saved.")
        else:
            patience_counter += 1
            log(f"  -> No improvement ({patience_counter}/{args.early_stop}), "
                f"best={best_mrr:.6f}")
            if patience_counter >= args.early_stop:
                log("Early stopping.")
                break

        # Save latest
        jt.save({
            "memory": memory.state_dict(),
            "gnn": gnn.state_dict(),
            "predictor": predictor.state_dict(),
        }, f"{save_path}/{args.dataset}_TGN_latest.pkl")
        jt.gc()

# ---- Load best model for inference ----
log("\nGenerating test predictions...")

# Replay all training edges to rebuild memory state
memory.reset_state()
neighbor_loader.reset_state()
memory.eval()
gnn.eval()
predictor.eval()

best_path = f"{save_path}/{args.dataset}_TGN_best.pkl"
latest_path = f"{save_path}/{args.dataset}_TGN_latest.pkl"
if args.model_path:
    load_path = args.model_path
elif osp.exists(best_path):
    load_path = best_path
elif osp.exists(latest_path):
    load_path = latest_path
else:
    load_path = None

if load_path:
    ckpt = jt.load(load_path)
    memory.load_state_dict(ckpt["memory"])
    gnn.load_state_dict(ckpt["gnn"])
    predictor.load_state_dict(ckpt["predictor"])
    log(f"Loaded model from {load_path}")

# Replay all training data (including val) to build full memory
log("Replaying all training edges to build memory state...")
all_src = df["src"].values.astype(np.int32)
all_dst = df["dst"].values.astype(np.int32)
all_t = df["time"].values.astype(np.float32)
n_all = len(all_src)
replay_batch = 1024
memory.reset_state()
neighbor_loader.reset_state()
for bi in tqdm(range(0, n_all, replay_batch), desc="Replay", ncols=100):
    be = min(bi + replay_batch, n_all)
    src_b = jt.Var(all_src[bi:be])
    dst_b = jt.Var(all_dst[bi:be])
    t_b = jt.Var(all_t[bi:be].astype(np.int32))
    msg_b = jt.zeros((be - bi, msg_dim))
    with jt.no_grad():
        memory.update_state(src_b, dst_b, t_b, msg_b)
        neighbor_loader.insert(src_b, dst_b)

log("Memory state built. Scoring test queries...")
scores = test_competition(
    memory, gnn, predictor, assoc, explicit,
    test_src, test_time, test_candidates,
    batch_size=100,
)

log(f"Scores shape: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
output_file = osp.join(args.output_dir, args.dataset, f"{args.dataset}_result_tgn.csv")
os.makedirs(osp.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {output_file}")
log("\n" + "=" * 80 + "\nDONE\n" + "=" * 80)
