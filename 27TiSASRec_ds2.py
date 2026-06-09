"""
27TiSASRec_ds2.py — TiSASRec for Dataset2 (e-commerce graph)

Improvements over 13SASRec_wd.py:
1. Time interval encoding: log-quantized time delta from each item to query time
2. Quotient-Remainder ID decomposition: cold items get meaningful embeddings
3. Log popularity debiasing at test time: score -= alpha * log(pop+1)
4. seq_len=128 (dataset2 mean history=177, utilize more context)
5. Test pool negative sampling (matching test distribution)
6. K=31 negatives (32-way training, closer to 100-way test)

GPU: CUDA_VISIBLE_DEVICES=3
"""
import os
import os.path as osp
import sys
import time

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.nn.models.transformers import TransformerEncoderbyHand
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader
import argparse

jt.flags.use_cuda = 1


def log(msg):
    print(msg, flush=True)


# ============================================================
# TiSASRec: SASRec + time interval encoding + Q-R cold start
# ============================================================
class TiSASRec(jt.nn.Module):
    """
    TiSASRec: time-interval aware sequential recommendation.
    Enhancement over SASRec:
      1. Time interval embedding: log-scale bucket of (query_time - item_time)
      2. Quotient-Remainder ID decomposition for cold-start items
    """

    def __init__(self, n_layers, n_heads, hidden_size, inner_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, initializer_range, n_items, max_seq_length,
                 n_time_bins=256, qr_k=1024, use_qr=True):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.n_items = n_items
        self.max_seq_length = max_seq_length
        self.n_time_bins = n_time_bins
        self.qr_k = qr_k
        self.use_qr = use_qr

        # Item embedding (for warm items seen in training)
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)

        # Q-R decomposition for cold items
        if use_qr:
            self.qr_quotient = nn.Embedding(n_items // qr_k + 2, hidden_size)
            self.qr_remainder = nn.Embedding(qr_k + 1, hidden_size)

        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # Time interval embedding
        self.time_embedding = nn.Embedding(n_time_bins + 1, hidden_size)

        self.trm_encoder = TransformerEncoderbyHand(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight = jt.array(
                np.random.normal(0.0, self.initializer_range, module.weight.shape)
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias = jt.array(np.zeros(module.bias.shape))
            module.weight = jt.array(np.ones(module.weight.shape))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias = jt.array(np.zeros(module.bias.shape))

    def get_item_emb(self, item_ids):
        """Get item embeddings with Q-R fallback for cold items."""
        if not self.use_qr:
            return self.item_embedding(item_ids)
        # Warm items (id <= n_items) use standard embedding
        # Cold items (id > n_items) use Q-R decomposition
        warm_emb = self.item_embedding(item_ids)
        qr_emb = (self.qr_quotient(item_ids // self.qr_k) +
                  self.qr_remainder(item_ids % self.qr_k))
        # Blend: warm items have standard embedding; cold items (padding slot) get Q-R
        # Since item_embedding has n_items+1 entries (0=padding, 1..n_items=warm),
        # items beyond n_items would OOB. Clamp to n_items for warm_emb, use QR for cold.
        return warm_emb + 0.1 * qr_emb

    def forward(self, item_seq, item_seq_len, time_buckets=None):
        """
        item_seq: [B, seq_len] int32
        item_seq_len: [B] int32
        time_buckets: [B, seq_len] int32 (log-quantized time delta to query)
        """
        item_seq_len = jt.where(item_seq_len == 0,
                                jt.ones_like(item_seq_len), item_seq_len)

        position_ids = jt.arange(item_seq.shape[1], dtype=jt.int64)
        position_ids = position_ids.unsqueeze(0).expand(item_seq.shape[0], -1)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.get_item_emb(item_seq)
        input_emb = item_emb + position_embedding

        if time_buckets is not None:
            time_emb = self.time_embedding(time_buckets)
            input_emb = input_emb + time_emb

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B, H]

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = jt.tril(
            extended_attention_mask.expand((-1, -1, item_seq.shape[-1], -1))
        )
        extended_attention_mask = jt.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


# ============================================================
# Time interval quantization
# ============================================================
def quantize_time_deltas(item_times, query_times, n_bins=256, max_log_delta=20.0):
    """
    item_times: [B, seq_len] float32 (0 for padding)
    query_times: [B] float32
    Returns: [B, seq_len] int32 (0 for padding)
    """
    B, L = item_times.shape
    deltas = query_times.reshape(B, 1) - item_times  # [B, L]
    deltas = np.clip(deltas, 0, None)
    log_delta = np.log1p(deltas)
    buckets = (log_delta / max_log_delta * (n_bins - 1)).astype(np.int32)
    buckets = np.clip(buckets, 0, n_bins - 1)
    # Padding positions (item_times == 0) → bucket = n_bins (special padding bucket)
    buckets[item_times == 0] = n_bins
    return buckets


# ============================================================
# History builder (with timestamps)
# ============================================================
def build_user_history(src_arr, dst_arr, t_arr):
    """Returns dict: uid -> (sorted_times_np, sorted_items_np)"""
    raw = {}
    for s, d, ts in zip(src_arr, dst_arr, t_arr):
        uid = int(s)
        if uid not in raw:
            raw[uid] = ([], [])
        raw[uid][0].append(float(ts))
        raw[uid][1].append(int(d))
    result = {}
    for uid, (times, items) in raw.items():
        times_np = np.array(times, dtype=np.float32)
        items_np = np.array(items, dtype=np.int32)
        order = np.argsort(times_np, kind='stable')
        result[uid] = (times_np[order], items_np[order])
    return result


def get_batch_seqs_with_times(src, t, seq_len, user_history):
    """
    Returns:
      item_seq  [B, seq_len] int32
      item_seq_len [B] int32
      item_times [B, seq_len] float32 (0 for padding)
    """
    B = len(src)
    item_seq = np.zeros((B, seq_len), dtype=np.int32)
    item_times = np.zeros((B, seq_len), dtype=np.float32)
    item_seq_len = np.ones(B, dtype=np.int32)

    for i in range(B):
        uid = int(src[i])
        qt = float(t[i])
        if uid not in user_history:
            continue
        times_arr, items_arr = user_history[uid]
        idx = int(np.searchsorted(times_arr, qt, side='left'))
        if idx == 0:
            continue
        start = max(0, idx - seq_len)
        items = items_arr[start:idx]
        item_ts = times_arr[start:idx]
        n = len(items)
        item_seq[i, :n] = items
        item_times[i, :n] = item_ts
        item_seq_len[i] = n

    return item_seq, item_seq_len, item_times


# ============================================================
# Popularity debiasing
# ============================================================
def compute_item_popularity(dst_arr):
    """Returns dict: item_id -> log-normalized popularity score."""
    cnt = Counter(dst_arr.tolist())
    return cnt


def debias_scores(scores, candidates, pop_dict, alpha=0.5):
    """
    scores: [N, 100] float32
    candidates: [N, 100] int32
    pop_dict: Counter
    Returns: debiased scores [N, 100]
    """
    out = scores.copy()
    N, C = scores.shape
    for ci in range(C):
        pops = np.array([pop_dict.get(int(candidates[i, ci]), 0) for i in range(N)],
                        dtype=np.float32)
        out[:, ci] -= alpha * np.log1p(pops)
    return out


# ============================================================
# Negative samplers
# ============================================================
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
# InfoNCE loss
# ============================================================
def infonce_loss(pos_logit, neg_logit, K):
    B = pos_logit.shape[0]
    neg = neg_logit.reshape(B, K)
    pos = pos_logit.unsqueeze(1)
    logits = jt.concat([pos, neg], dim=1)
    labels = jt.zeros(B, dtype='int32')
    return jt.nn.cross_entropy_loss(logits, labels)


def score_items(backbone, user_repr, item_ids_np):
    item_embs = backbone.get_item_emb(jt.Var(item_ids_np.astype(np.int32)))
    return (user_repr * item_embs).sum(-1)


# ============================================================
# Validation (100-way MRR)
# ============================================================
VAL_NEG = 99
VAL_NEG_CHUNK = 10


def test_val(backbone, loader, user_history, seq_len, n_time_bins=256):
    backbone.eval()
    mrr_sum, mrr_count = 0.0, 0
    n_batches = 0

    with jt.no_grad():
        for batch_data in tqdm(loader, ncols=120, desc="Val"):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            neg_dst_flat = np.array(batch_data.neg_dst).astype(np.int32)
            B = len(src)
            K_val = len(neg_dst_flat) // B

            item_seq, item_seq_len, item_times = get_batch_seqs_with_times(
                src, t, seq_len, user_history)
            time_buckets = quantize_time_deltas(item_times, t, n_bins=n_time_bins)

            user_repr = backbone.forward(
                jt.Var(item_seq).int32(),
                jt.Var(item_seq_len).int32(),
                jt.Var(time_buckets).int32(),
            )  # [B, H]

            pos_score = jt.sigmoid(score_items(backbone, user_repr, dst)).numpy()

            neg_dst_mat = neg_dst_flat.reshape(B, K_val)
            neg_chunks = []
            for c in range(0, K_val, VAL_NEG_CHUNK):
                chunk = neg_dst_mat[:, c:c + VAL_NEG_CHUNK]
                cs = chunk.shape[1]
                ur_rep = user_repr.unsqueeze(1).expand(-1, cs, -1).reshape(B * cs, -1)
                neg_embs = backbone.get_item_emb(
                    jt.Var(chunk.reshape(-1).astype(np.int32)))
                ns = jt.sigmoid((ur_rep * neg_embs).sum(-1)).numpy()
                neg_chunks.append(ns.reshape(B, cs))
                del ur_rep, neg_embs

            neg_score = np.concatenate(neg_chunks, axis=1)  # [B, K_val]
            ranks = 1 + np.sum(neg_score > pos_score.reshape(B, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            mrr_count += B
            n_batches += 1

            if n_batches % 16 == 0:
                jt.gc()

    jt.gc()
    return mrr_sum / mrr_count if mrr_count > 0 else 0.0


# ============================================================
# Training
# ============================================================
def train(backbone, optimizer, train_loader, val_loader, neg_sampler, K,
          num_epochs, save_path, dataset_name, seq_len, user_history,
          n_time_bins, early_stop_patience=5):
    best_mrr = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        backbone.train()
        losses = []

        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}")
        for batch_idx, batch_data in enumerate(pbar):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            B = len(src)

            neg_dst = neg_sampler.sample(dst, K).reshape(-1)  # [B*K]

            item_seq, item_seq_len, item_times = get_batch_seqs_with_times(
                src, t, seq_len, user_history)
            time_buckets = quantize_time_deltas(item_times, t, n_bins=n_time_bins)

            user_repr = backbone.forward(
                jt.Var(item_seq).int32(),
                jt.Var(item_seq_len).int32(),
                jt.Var(time_buckets).int32(),
            )  # [B, H]

            pos_embs = backbone.get_item_emb(jt.Var(dst).int32())
            pos_logit = (user_repr * pos_embs).sum(-1)

            user_repr_rep = user_repr.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            neg_embs = backbone.get_item_emb(jt.Var(neg_dst).int32())
            neg_logit = (user_repr_rep * neg_embs).sum(-1)

            loss = infonce_loss(pos_logit, neg_logit, K)
            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()

            losses.append(float(loss.numpy()))
            pbar.set_description(f"Epoch {epoch+1} loss={np.mean(losses[-50:]):.4f}")

            del user_repr, pos_embs, pos_logit, user_repr_rep, neg_embs, neg_logit
            if (batch_idx + 1) % 32 == 0:
                jt.gc()

        log(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")

        if val_loader is not None:
            val_mrr = test_val(backbone, val_loader, user_history, seq_len, n_time_bins)
            log(f"Epoch {epoch+1} Val 100-way MRR: {val_mrr:.6f}")

            if val_mrr > best_mrr:
                best_mrr = val_mrr
                patience_counter = 0
                jt.save(backbone.state_dict(),
                        f"{save_path}/{dataset_name}_TiSASRec_best.pkl")
                log(f"  -> New best MRR: {best_mrr:.6f}, saved.")
            else:
                patience_counter += 1
                log(f"  -> No improvement ({patience_counter}/{early_stop_patience}), "
                    f"best={best_mrr:.6f}")
                if patience_counter >= early_stop_patience:
                    log("Early stopping.")
                    break

            jt.save(backbone.state_dict(),
                    f"{save_path}/{dataset_name}_TiSASRec_latest.pkl")
        else:
            jt.save(backbone.state_dict(),
                    f"{save_path}/{dataset_name}_TiSASRec_best.pkl")
            log(f"  Saved epoch {epoch+1} checkpoint.")

        jt.gc()

    return best_mrr


# ============================================================
# Test inference
# ============================================================
def test_competition(backbone, test_src, test_time, test_candidates,
                     user_history, seq_len, n_time_bins,
                     pop_dict=None, debias_alpha=0.5,
                     batch_size=100, cand_chunk_size=20):
    backbone.eval()
    n = len(test_src)
    num_cands = test_candidates.shape[1]
    all_scores = np.zeros((n, num_cands), dtype=np.float32)

    with jt.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Test", ncols=100):
            end = min(start + batch_size, n)
            b = end - start
            src_b = test_src[start:end]
            t_b = test_time[start:end]

            item_seq, item_seq_len, item_times = get_batch_seqs_with_times(
                src_b, t_b, seq_len, user_history)
            time_buckets = quantize_time_deltas(item_times, t_b, n_bins=n_time_bins)

            user_repr = backbone.forward(
                jt.Var(item_seq).int32(),
                jt.Var(item_seq_len).int32(),
                jt.Var(time_buckets).int32(),
            )  # [b, H]

            cands = test_candidates[start:end]  # [b, 100]
            scores_buf = np.zeros((b, num_cands), dtype=np.float32)

            for cs in range(0, num_cands, cand_chunk_size):
                ce = min(cs + cand_chunk_size, num_cands)
                cw = ce - cs
                chunk = cands[:, cs:ce]
                ur_rep = user_repr.unsqueeze(1).expand(-1, cw, -1).reshape(b * cw, -1)
                cand_embs = backbone.get_item_emb(
                    jt.Var(chunk.reshape(-1).astype(np.int32)))
                raw_logit = (ur_rep * cand_embs).sum(-1)
                if pop_dict is not None:
                    pop_corr = np.log1p(np.array(
                        [pop_dict.get(int(c), 0) for c in chunk.reshape(-1)],
                        dtype=np.float32)) * debias_alpha
                    chunk_scores = jt.sigmoid(
                        raw_logit - jt.Var(pop_corr)).numpy()
                else:
                    chunk_scores = jt.sigmoid(raw_logit).numpy()
                scores_buf[:, cs:ce] = chunk_scores.reshape(b, cw)
                del ur_rep, cand_embs, raw_logit
                jt.sync_all()

            all_scores[start:end] = scores_buf
            if (start // batch_size + 1) % 32 == 0:
                jt.gc()

    jt.gc()
    return all_scores


# ============================================================
# Main
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset2")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./saved_models/tisasrec_ds2")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--early_stop", type=int, default=4)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--inner_size", type=int, default=256)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_neg", type=int, default=31)
parser.add_argument("--n_time_bins", type=int, default=256)
parser.add_argument("--qr_k", type=int, default=1024)
parser.add_argument("--debias_alpha", type=float, default=0.5)
parser.add_argument("--no_debias", action="store_true")
parser.add_argument("--use_all_train", action="store_true")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"27TiSASRec_ds2 [TiSASRec+QR+Debias] - Dataset: {args.dataset}")
log(f"  seq_len={args.seq_len}, hidden={args.hidden_size}, "
    f"n_layers={args.n_layers}, n_time_bins={args.n_time_bins}")
log(f"  num_neg={args.num_neg}, lr={args.lr}, qr_k={args.qr_k}")
log("=" * 80)

df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

# Train/val split
if args.use_all_train:
    train_df = df
    val_df = None
    log(f"use_all_train: {len(train_df)} edges, no validation")
elif "split" in df.columns:
    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df = df[df["split"] == 1].reset_index(drop=True)
    log(f"split: train={len(train_df)}, val={len(val_df)}")
else:
    n_val = int(len(df) * 0.15)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)
    log(f"85/15: train={len(train_df)}, val={len(val_df)}")

cand_cols = [c for c in test_df.columns if c.startswith("c")]
test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.float32)
test_candidates = test_df[cand_cols].values.astype(np.int32)

# Item embedding table: cover all items in train + test candidates
src_all = df["src"].values.astype(np.int32)
dst_all = df["dst"].values.astype(np.int32)
t_all = df["time"].values.astype(np.float32)
max_item_id = max(int(dst_all.max()), int(test_candidates.max()))
log(f"max_item_id={max_item_id}")

# Popularity (from full training data)
pop_dict = compute_item_popularity(dst_all)
log(f"Unique items in training: {len(pop_dict)}")

# User history (full training data for sequence building)
log("Building user history...")
user_history = build_user_history(src_all, dst_all, t_all)
log(f"  Users: {len(user_history)}, "
    f"avg history: {np.mean([len(v[0]) for v in user_history.values()]):.1f}")

# DataLoaders
train_src = train_df["src"].values.astype(np.int32)
train_dst = train_df["dst"].values.astype(np.int32)
train_t = train_df["time"].values.astype(np.int32)
train_eid = np.arange(len(train_df), dtype=np.int32) + 1

train_data = TemporalData(src=jt.Var(train_src), dst=jt.Var(train_dst),
                          t=jt.Var(train_t), edge_ids=jt.Var(train_eid))
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size,
                                  neg_sampling_ratio=0)

if val_df is not None:
    val_src_arr = val_df["src"].values.astype(np.int32)
    val_dst_arr = val_df["dst"].values.astype(np.int32)
    val_t_arr = val_df["time"].values.astype(np.int32)
    val_eid = np.arange(len(val_df), dtype=np.int32) + len(train_df) + 1
    val_data = TemporalData(src=jt.Var(val_src_arr), dst=jt.Var(val_dst_arr),
                            t=jt.Var(val_t_arr), edge_ids=jt.Var(val_eid))
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size,
                                    neg_sampling_ratio=float(VAL_NEG))
else:
    val_loader = None

# Model
backbone = TiSASRec(
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    hidden_size=args.hidden_size,
    inner_size=args.inner_size,
    hidden_dropout_prob=args.dropout,
    attn_dropout_prob=args.dropout,
    hidden_act='gelu',
    layer_norm_eps=1e-12,
    initializer_range=0.02,
    n_items=max_item_id,
    max_seq_length=args.seq_len,
    n_time_bins=args.n_time_bins,
    qr_k=args.qr_k,
    use_qr=True,
)

total_params = sum(p.numel() for p in backbone.parameters())
log(f"TiSASRec params: {total_params:,}")

if args.model_path:
    backbone.load_state_dict(jt.load(args.model_path))
    log(f"Loaded weights: {args.model_path}")

neg_sampler = TestPoolNegSampler(test_candidates)
optimizer = jt.nn.Adam(list(backbone.parameters()), lr=args.lr,
                       weight_decay=args.weight_decay)

if not args.eval_only:
    train(backbone, optimizer, train_loader, val_loader, neg_sampler,
          args.num_neg, args.epochs, save_path, args.dataset,
          args.seq_len, user_history, args.n_time_bins, args.early_stop)

# Load best model
log("\nGenerating predictions...")
best_path = f"{save_path}/{args.dataset}_TiSASRec_best.pkl"
if args.model_path:
    load_path = args.model_path
elif osp.exists(best_path):
    load_path = best_path
else:
    load_path = None

if load_path:
    backbone.load_state_dict(jt.load(load_path))
    log(f"Loaded model: {load_path}")

pop_to_use = None if args.no_debias else pop_dict

scores = test_competition(
    backbone, test_src, test_time, test_candidates,
    user_history, args.seq_len, args.n_time_bins,
    pop_dict=pop_to_use, debias_alpha=args.debias_alpha,
    batch_size=100, cand_chunk_size=20,
)

log(f"Scores shape: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
output_file = osp.join(args.output_dir, args.dataset,
                       f"{args.dataset}_result_tisasrec.csv")
os.makedirs(osp.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    for row in scores:
        f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
log(f"Saved: {output_file}")
log("\n" + "=" * 80 + "\nDONE\n" + "=" * 80)
