"""
16SASRec_inbatch.py — In-batch Negative Sampling + SASRec

Key insight: test negatives score 0.963 (model's own high scores = hard negatives).
Training with random negatives never teaches the model to handle this.

Fix: Use ALL other positives in the batch as negatives (in-batch contrastive).
- In-batch negatives are naturally hard (popular items click by many users).
- Reduces popularity bias: popular items appear as negatives frequently.
- Temperature τ=0.1 sharpens the discrimination signal.
- Also keeps K=7 sampled negatives for diversity.

vs 13SASRec_wd.py changes:
- infonce_inbatch(): uses B×B similarity matrix + temperature
- seq_len=128 default (was 64)
- temperature=0.1 default
- mixed loss: in-batch + 0.3 × sampled
"""
import os
import os.path as osp
import sys
import time
from collections import Counter

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.nn.models.sasrec import SASRec
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader
from jittor_geometric.data import TemporalData
import argparse

jt.flags.use_cuda = 1
IS_MPI = jt.in_mpi
RANK = jt.rank if IS_MPI else 0
WORLD_SIZE = jt.world_size if IS_MPI else 1
IS_MAIN = RANK == 0


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


def build_user_history(src_arr, dst_arr, t_arr):
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


def get_batch_seqs(src, t, seq_len, user_history):
    B = len(src)
    item_seq = np.zeros((B, seq_len), dtype=np.int32)
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
        n = len(items)
        item_seq[i, :n] = items
        item_seq_len[i] = n

    return item_seq, item_seq_len


class PopNegSampler:
    def __init__(self, df, p_pop=0.7, seed=42):
        self.p_pop = p_pop
        self._rng = np.random.RandomState(seed)
        dst_vals = df["dst"].values
        self.min_dst = int(dst_vals.min())
        self.max_dst = int(dst_vals.max())

        self.src_history = {}
        for s, d in zip(df["src"].values, dst_vals):
            self.src_history.setdefault(int(s), set()).add(int(d))

        dst_counts = Counter(dst_vals.tolist())
        dsts = np.array(list(dst_counts.keys()), dtype=np.int32)
        freqs = np.array([dst_counts[d] for d in dsts], dtype=np.float64)
        sqrt_freqs = np.sqrt(freqs)
        self.pop_dsts = dsts
        self.pop_probs = sqrt_freqs / sqrt_freqs.sum()

    def sample(self, src, dst, K):
        B = len(src)
        n_pop = max(1, int(K * self.p_pop))
        result = np.empty(B * K, dtype=np.int32)

        for i, (s, d) in enumerate(zip(src, dst)):
            hist = self.src_history.get(int(s), set())
            cands = []

            over = self._rng.choice(self.pop_dsts, size=n_pop * 5, p=self.pop_probs, replace=True)
            for c in over:
                if int(c) != int(d) and int(c) not in hist:
                    cands.append(int(c))
                    if len(cands) >= n_pop:
                        break

            tries = 0
            while len(cands) < K and tries < K * 10:
                c = self._rng.randint(self.min_dst, self.max_dst + 1)
                tries += 1
                if c != int(d) and c not in hist:
                    cands.append(c)

            while len(cands) < K:
                cands.append(self._rng.randint(self.min_dst, self.max_dst + 1))

            result[i * K:(i + 1) * K] = cands[:K]

        return result


def infonce_inbatch(user_repr, pos_embs, temperature=0.1):
    """
    In-batch contrastive loss.
    user_repr: [B, H], pos_embs: [B, H]
    sim[i,j] = dot(user_i, item_j) / temperature
    loss = cross_entropy(sim, arange(B))  # positive is diagonal
    """
    B = user_repr.shape[0]
    sim = jt.matmul(user_repr, pos_embs.transpose(0, 1)) / temperature  # [B, B]
    labels = jt.zeros(B, dtype='int32')  # placeholder, actual below
    labels = jt.array(np.arange(B, dtype=np.int32))
    return jt.nn.cross_entropy_loss(sim, labels)


def infonce_sampled(pos_logit, neg_logit, K):
    B = pos_logit.shape[0]
    neg = neg_logit.reshape(B, K)
    pos = pos_logit.unsqueeze(1)
    logits = jt.concat([pos, neg], dim=1)
    labels = jt.zeros(B, dtype='int32')
    return jt.nn.cross_entropy_loss(logits, labels)


def score_items(backbone, user_repr, item_ids_np):
    item_embs = backbone.item_embedding(jt.Var(item_ids_np.astype(np.int32)))
    return (user_repr * item_embs).sum(-1)


VAL_NEG = 99
VAL_NEG_CHUNK = 10


def test_val(backbone, loader, user_history, seq_len):
    backbone.eval()
    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count = 0.0, 0

    with jt.no_grad():
        for batch_idx, batch_data in enumerate(
                tqdm(loader, ncols=120, desc="Val", disable=not IS_MAIN)):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            neg_dst_flat = np.array(batch_data.neg_dst).astype(np.int32)
            B = len(src)
            K_val = len(neg_dst_flat) // B

            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = backbone.forward(
                jt.Var(item_seq).int32(),
                jt.Var(item_seq_len).int32()
            )

            pos_score = jt.sigmoid(score_items(backbone, user_repr, dst)).numpy()

            neg_dst_mat = neg_dst_flat.reshape(B, K_val)
            neg_chunks = []
            for c in range(0, K_val, VAL_NEG_CHUNK):
                chunk = neg_dst_mat[:, c:c + VAL_NEG_CHUNK]
                cs = chunk.shape[1]
                user_repr_rep = user_repr.unsqueeze(1).expand(-1, cs, -1).reshape(B * cs, -1)
                neg_embs = backbone.item_embedding(
                    jt.Var(chunk.reshape(-1).astype(np.int32)))
                ns = jt.sigmoid((user_repr_rep * neg_embs).sum(-1)).numpy()
                neg_chunks.append(ns.reshape(B, cs))
                del user_repr_rep, neg_embs

            neg_score = np.concatenate(neg_chunks, axis=1)

            y_true = np.concatenate([np.ones(B), np.zeros(B * K_val)])
            y_score = np.concatenate([pos_score, neg_score.reshape(-1)])
            ap_sum += float(average_precision_score(y_true, y_score))
            auc_sum += float(roc_auc_score(y_true, y_score))
            n_batches += 1

            ranks = 1 + np.sum(neg_score > pos_score.reshape(B, 1), axis=1)
            mrr_sum += float(np.sum(1.0 / ranks))
            mrr_count += B

            if (batch_idx + 1) % 16 == 0:
                jt.gc()
            jt.sync_all()

    jt.gc()
    if n_batches == 0:
        return {"AP": 0.0, "AUC": 0.0, "MRR": 0.0}
    return {
        "AP": ap_sum / n_batches,
        "AUC": auc_sum / n_batches,
        "MRR": mrr_sum / mrr_count,
    }


def train(backbone, optimizer, train_loader, val_loader, neg_sampler, K,
          num_epochs, save_path, dataset_name, seq_len, user_history,
          temperature, sampled_weight, early_stop_patience=5):
    best_mrr = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        backbone.train()
        losses = []

        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch + 1}", disable=not IS_MAIN)
        for batch_idx, batch_data in enumerate(pbar):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            B = len(src)

            item_seq, item_seq_len = get_batch_seqs(src, t, seq_len, user_history)
            user_repr = backbone.forward(
                jt.Var(item_seq).int32(),
                jt.Var(item_seq_len).int32()
            )  # [B, H]

            pos_embs = backbone.item_embedding(jt.Var(dst).int32())  # [B, H]

            # Primary: in-batch contrastive loss
            loss_ib = infonce_inbatch(user_repr, pos_embs, temperature)

            # Auxiliary: sampled negatives
            if K > 0 and sampled_weight > 0:
                neg_dst = neg_sampler.sample(src, dst, K)
                pos_logit = (user_repr * pos_embs).sum(-1)
                user_repr_rep = user_repr.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
                neg_embs = backbone.item_embedding(jt.Var(neg_dst).int32())
                neg_logit = (user_repr_rep * neg_embs).sum(-1)
                loss_sampled = infonce_sampled(pos_logit, neg_logit, K)
                loss = loss_ib + sampled_weight * loss_sampled
                del user_repr_rep, neg_embs, neg_logit
            else:
                loss = loss_ib

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()

            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch + 1} loss={loss.item():.4f}")

            del user_repr, pos_embs
            if (batch_idx + 1) % 32 == 0:
                jt.gc()

        log(f"Epoch {epoch + 1} Train Loss: {np.mean(losses):.4f}")

        if val_loader is not None:
            val_res = test_val(backbone, val_loader, user_history, seq_len)
            log(f"Epoch {epoch + 1} Val: {val_res}")
            cur = val_res["MRR"]
            if cur > best_mrr:
                best_mrr = cur
                patience_counter = 0
                if IS_MAIN:
                    jt.save(backbone.state_dict(),
                            f"{save_path}/{dataset_name}_SASRec_best.pkl")
                    os.sync()
                log(f"  -> New best MRR: {best_mrr:.6f}")
            else:
                patience_counter += 1
                log(f"  -> No improvement ({patience_counter}/{early_stop_patience}), best={best_mrr:.6f}")
            if IS_MAIN:
                jt.save(backbone.state_dict(),
                        f"{save_path}/{dataset_name}_SASRec.pkl")
                os.sync()
            if patience_counter >= early_stop_patience:
                log("Early stopping.")
                break
        else:
            if IS_MAIN:
                ckpt = f"{save_path}/{dataset_name}_SASRec_epoch{epoch+1}.pkl"
                jt.save(backbone.state_dict(), ckpt)
                jt.save(backbone.state_dict(),
                        f"{save_path}/{dataset_name}_SASRec_best.pkl")
                jt.save(backbone.state_dict(),
                        f"{save_path}/{dataset_name}_SASRec.pkl")
                log(f"  Saved epoch {epoch+1} checkpoint: {ckpt}")
                os.sync()

        jt.gc()
    return best_mrr


def test_competition(backbone, test_src, test_time, test_candidates,
                     save_path, dataset_name, user_history, seq_len,
                     batch_size=50, cand_chunk_size=20):
    backbone.eval()
    n = len(test_src)
    num_cands = test_candidates.shape[1]
    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n = len(my_idx)
    my_scores = np.zeros((my_n, num_cands), dtype=np.float32)

    pbar = tqdm(range((my_n + batch_size - 1) // batch_size),
                ncols=120, desc=f"Testing[r{RANK}]")
    with jt.no_grad():
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, my_n)
            idx_chunk = my_idx[start:end]
            b = len(idx_chunk)

            src_b = test_src[idx_chunk]
            t_b = test_time[idx_chunk].astype(np.float32)

            item_seq, item_seq_len = get_batch_seqs(src_b, t_b, seq_len, user_history)
            user_repr = backbone.forward(
                jt.Var(item_seq).int32(),
                jt.Var(item_seq_len).int32()
            )

            cands = test_candidates[idx_chunk]
            scores_buf = np.empty((b, num_cands), dtype=np.float32)

            for cs in range(0, num_cands, cand_chunk_size):
                ce = min(cs + cand_chunk_size, num_cands)
                chunk_size = ce - cs
                chunk = cands[:, cs:ce]
                user_repr_rep = user_repr.unsqueeze(1).expand(
                    -1, chunk_size, -1).reshape(b * chunk_size, -1)
                cand_embs = backbone.item_embedding(
                    jt.Var(chunk.reshape(-1).astype(np.int32)))
                chunk_scores = jt.sigmoid(
                    (user_repr_rep * cand_embs).sum(-1)).numpy()
                scores_buf[:, cs:ce] = chunk_scores.reshape(b, chunk_size)
                del user_repr_rep, cand_embs
                jt.sync_all()

            my_scores[start:end] = scores_buf
            if (batch_idx + 1) % 64 == 0:
                jt.gc()

    jt.gc()

    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    np.save(f"{save_path}/_tmp_{dataset_name}_rank{RANK}_idx.npy", my_idx)
    np.save(f"{save_path}/_tmp_{dataset_name}_rank{RANK}_scores.npy", my_scores)
    np.save(f"{save_path}/_tmp_{dataset_name}_rank{RANK}_done.npy",
            np.array([1], dtype=np.int8))

    def _wait(path, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            time.sleep(1)
        return False

    missing = [r for r in range(WORLD_SIZE)
               if not _wait(f"{save_path}/_tmp_{dataset_name}_rank{r}_done.npy")]
    if missing:
        print(f"[r{RANK}] missing ranks {missing}")
        return None
    if not IS_MAIN:
        return None

    all_idx = np.concatenate([
        np.load(f"{save_path}/_tmp_{dataset_name}_rank{r}_idx.npy")
        for r in range(WORLD_SIZE)])
    all_sc = np.concatenate([
        np.load(f"{save_path}/_tmp_{dataset_name}_rank{r}_scores.npy")
        for r in range(WORLD_SIZE)])
    result = all_sc[np.argsort(all_idx)]
    for r in range(WORLD_SIZE):
        for suf in ["_idx.npy", "_scores.npy", "_done.npy"]:
            try:
                os.remove(f"{save_path}/_tmp_{dataset_name}_rank{r}{suf}")
            except Exception:
                pass
    return result


# ============================================================
# Main
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./saved_models")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--inner_size", type=int, default=256)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_neg", type=int, default=7,
                    help="Sampled negatives (auxiliary, in addition to in-batch)")
parser.add_argument("--sampled_weight", type=float, default=0.3,
                    help="Weight for sampled negative loss (0=in-batch only)")
parser.add_argument("--p_pop", type=float, default=0.7)
parser.add_argument("--temperature", type=float, default=0.1,
                    help="Temperature for in-batch contrastive loss")
parser.add_argument("--use_all_train", action="store_true")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--test_batch_size", type=int, default=50)
parser.add_argument("--cand_chunk_size", type=int, default=20)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"16SASRec_inbatch [In-batch NCE + WeightDecay] - Dataset: {args.dataset}")
log(f"  seq_len={args.seq_len}, hidden={args.hidden_size}, "
    f"n_layers={args.n_layers}, n_heads={args.n_heads}")
log(f"  temperature={args.temperature}, num_neg={args.num_neg}, "
    f"sampled_weight={args.sampled_weight}, lr={args.lr}")
log(f"  use_all_train={args.use_all_train}")
log("=" * 80)

df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

if args.use_all_train:
    train_df = df
    val_df = None
    log(f"use_all_train: {len(train_df)} 行，无验证集")
elif "split" in df.columns:
    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df = df[df["split"] == 1].reset_index(drop=True)
    log(f"split 列: train={len(train_df)}, val={len(val_df)}")
else:
    n_val = int(len(df) * 0.15)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)
    log(f"85/15 split: train={len(train_df)}, val={len(val_df)}")

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.float32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

src_all = df["src"].values.astype(np.int32)
dst_all = df["dst"].values.astype(np.int32)
t_all = df["time"].values.astype(np.float32)

max_item_id = max(int(dst_all.max()), int(test_candidates.max()))
log(f"max_item_id={max_item_id}  (embedding table size={max_item_id + 1})")

log("Building user history...")
user_history = build_user_history(src_all, dst_all, t_all)
log(f"  Users with history: {len(user_history)}, "
    f"avg history len: {np.mean([len(v[0]) for v in user_history.values()]):.1f}")

train_src = train_df["src"].values.astype(np.int32)
train_dst = train_df["dst"].values.astype(np.int32)
train_t = train_df["time"].values.astype(np.int32)
train_eid = np.arange(len(train_df), dtype=np.int32) + 1

train_data = TemporalData(src=jt.Var(train_src), dst=jt.Var(train_dst),
                          t=jt.Var(train_t), edge_ids=jt.Var(train_eid))
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size,
                                  neg_sampling_ratio=0)

if val_df is not None:
    val_src = val_df["src"].values.astype(np.int32)
    val_dst = val_df["dst"].values.astype(np.int32)
    val_t = val_df["time"].values.astype(np.int32)
    val_eid = np.arange(len(val_df), dtype=np.int32) + len(train_df) + 1
    val_data = TemporalData(src=jt.Var(val_src), dst=jt.Var(val_dst),
                            t=jt.Var(val_t), edge_ids=jt.Var(val_eid))
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size,
                                    neg_sampling_ratio=float(VAL_NEG))
else:
    val_loader = None

log(f"Node/edge info: {len(user_history)} users, max_item={max_item_id}, "
    f"train_edges={len(train_df)}")

backbone = SASRec(
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
)
backbone.set_min_idx(user_min_idx=0, item_min_idx=0)

log(f"SASRec params: {sum(p.numel() for p in backbone.parameters()):,}")

if args.resume_path and not args.eval_only:
    backbone.load_state_dict(jt.load(args.resume_path))
    log(f"Resumed weights from: {args.resume_path}")

if not args.eval_only:
    neg_sampler = PopNegSampler(train_df, p_pop=args.p_pop) if args.num_neg > 0 else None
    optimizer = jt.nn.Adam(list(backbone.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    log(f"\nIn-batch NCE: temperature={args.temperature}, "
        f"K_sampled={args.num_neg}, sampled_weight={args.sampled_weight}")
    train(backbone, optimizer, train_loader, val_loader, neg_sampler,
          args.num_neg, args.epochs, save_path, args.dataset,
          args.seq_len, user_history, args.temperature, args.sampled_weight,
          args.early_stop)
else:
    log("\n--eval_only: skip training.")

log("\nGenerating predictions...")
if args.model_path:
    backbone.load_state_dict(jt.load(args.model_path))
    log(f"Loaded: {args.model_path}")
else:
    best_path = f"{save_path}/{args.dataset}_SASRec_best.pkl"
    last_path = f"{save_path}/{args.dataset}_SASRec.pkl"
    if os.path.exists(best_path):
        backbone.load_state_dict(jt.load(best_path))
        log(f"Loaded best: {best_path}")
    elif os.path.exists(last_path):
        backbone.load_state_dict(jt.load(last_path))
        log("Loaded latest model.")
    else:
        log("No saved model found, using random weights.")

scores = test_competition(
    backbone, test_src, test_time, test_candidates,
    save_path, args.dataset, user_history, args.seq_len,
    args.test_batch_size, args.cand_chunk_size,
)

if IS_MAIN:
    if scores is None:
        sys.exit(0)
    log(f"Scores shape: {scores.shape}, "
        f"range=[{scores.min():.4f}, {scores.max():.4f}]")
    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result_sasrec_wd.csv"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for row in scores:
            f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    log(f"Saved: {output_file}")

log("\n" + "=" * 80 + "\nDONE\n" + "=" * 80)
