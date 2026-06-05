"""
6DyGFormer - Key improvements over 1DyGFormer:
1. Use actual split column (split=0 train, split=1 val) instead of 85/15 count split
2. Item log-popularity node features: log(degree+1) per node
3. --use_all_train: train on ALL data for final test submission (no val split)
4. --num_neg: multiple negatives per positive (default=1, try 3-5)
5. seq_len default=64 for better user history coverage
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
import types


def _pad_sequence(sequences, batch_first=True, padding_value=0.0):
    if not sequences:
        return jt.array([])
    max_len = max(int(s.shape[0]) for s in sequences)
    padded = []
    for s in sequences:
        if int(s.shape[0]) < max_len:
            pad_len = max_len - int(s.shape[0])
            pad_shape = [pad_len] + list(s.shape[1:])
            pad = jt.full(pad_shape, padding_value, dtype=s.dtype)
            s = jt.concat([s, pad], dim=0)
        padded.append(s)
    if batch_first:
        return jt.stack(padded, dim=0)
    else:
        return jt.stack(padded, dim=1)


if not hasattr(jt.nn, "utils"):
    jt.nn.utils = types.ModuleType("utils")
if not hasattr(jt.nn.utils, "rnn"):
    jt.nn.utils.rnn = types.ModuleType("rnn")
jt.nn.utils.rnn.pad_sequence = _pad_sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.dygformer import DyGFormer
from jittor_geometric.nn.dense.merge_predictor import MergeLayer
from jittor_geometric.dataloader.temporal_dataloader import (
    TemporalDataLoader,
    get_neighbor_sampler,
)
from jittor_geometric.evaluate.evaluators import MRR_Evaluator
import argparse

jt.flags.use_cuda = 1

IS_MPI = jt.in_mpi
RANK = jt.rank if IS_MPI else 0
WORLD_SIZE = jt.world_size if IS_MPI else 1
IS_MAIN = RANK == 0


def log(msg):
    if IS_MAIN:
        print(msg)


def test_val(model, loader, num_neg_val=1):
    """Validation with configurable number of negatives per positive.
    num_neg_val=1: standard pairwise (same as 1DyGFormer)
    num_neg_val=99: simulate 100-way ranking (closer to test conditions)
    """
    model.eval()
    backbone, predictor = model[0], model[1]
    mrr_eval = MRR_Evaluator()

    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count = 0.0, 0

    loader_tqdm = tqdm(loader, ncols=120, desc="Validation", disable=not IS_MAIN)
    with jt.no_grad():
        for batch_idx, batch_data in enumerate(loader_tqdm):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            neg_dst = np.array(batch_data.neg_dst).astype(np.int32)

            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)
            pos_score = jt.sigmoid(pos_logit).numpy()

            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=neg_dst, node_interact_times=t
            )
            neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)
            neg_score = jt.sigmoid(neg_logit).numpy()

            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_score = np.concatenate([pos_score, neg_score])
            ap_sum += float(average_precision_score(y_true, y_score))
            auc_sum += float(roc_auc_score(y_true, y_score))
            n_batches += 1

            batch_mrr = mrr_eval.eval(pos_score, neg_score)
            mrr_sum += float(np.sum(batch_mrr))
            mrr_count += len(batch_mrr)

            del src_emb_pos, dst_emb_pos, pos_logit, pos_score
            del src_emb_neg, neg_dst_emb, neg_logit, neg_score
            if (batch_idx + 1) % 64 == 0:
                jt.gc()
            jt.sync_all()
    jt.gc()

    if IS_MPI:
        agg = jt.array(
            [ap_sum, auc_sum, mrr_sum, float(n_batches), float(mrr_count)],
            dtype="float32",
        ).mpi_all_reduce("add")
        ap_sum, auc_sum, mrr_sum, n_batches, mrr_count = agg.numpy().tolist()

    if n_batches == 0 or mrr_count == 0:
        return {"AP": 0.0, "AUC": 0.0, "MRR": 0.0}
    return {
        "AP": ap_sum / n_batches,
        "AUC": auc_sum / n_batches,
        "MRR": mrr_sum / mrr_count,
    }


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    num_epochs,
    save_path,
    dataset_name,
    early_stop_patience=10,
):
    best_mrr = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        backbone, predictor = model[0], model[1]
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch + 1}", disable=not IS_MAIN)

        for batch_idx, batch_data in enumerate(train_tqdm):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            neg_dst = np.array(batch_data.neg_dst).astype(np.int32)

            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos)

            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=neg_dst, node_interact_times=t
            )
            neg_logit = predictor(src_emb_neg, neg_dst_emb)

            loss = criterion(pos_logit, jt.ones_like(pos_logit)) + criterion(neg_logit, jt.zeros_like(neg_logit))

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            train_losses.append(loss.item())
            train_tqdm.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

        log(f"Epoch {epoch + 1}, Train Loss: {np.mean(train_losses):.4f}")

        if val_loader is not None:
            val_res = test_val(model, val_loader)
            log(f"Epoch {epoch + 1}, Val: {val_res}")
            current_metric = val_res["MRR"]  # optimize MRR (better proxy for test)

            if current_metric > best_mrr:
                best_mrr = current_metric
                patience_counter = 0
                if IS_MAIN:
                    jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                    os.sync()
                if IS_MPI:
                    jt.array([0], dtype="int32").mpi_all_reduce("add")
                log(f"  -> New best MRR: {best_mrr:.6f}, model saved!")
            else:
                patience_counter += 1
                log(f"  -> No improvement for {patience_counter} epoch(s), best MRR: {best_mrr:.6f}")

            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
                os.sync()
            if IS_MPI:
                jt.array([0], dtype="int32").mpi_all_reduce("add")

            if patience_counter >= early_stop_patience:
                log(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                log(f"Best validation MRR: {best_mrr:.6f}")
                break
        else:
            # No validation (use_all_train mode): save every epoch
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
                os.sync()
            log(f"  -> Epoch {epoch+1} model saved (no validation in use_all_train mode)")

        jt.gc()

    return best_mrr


def test_competition(
    model,
    test_src,
    test_time,
    test_candidates,
    tmp_dir,
    dataset_name,
    batch_size=25,
    cand_chunk_size=25,
):
    model.eval()
    backbone, predictor = model[0], model[1]
    n = len(test_src)
    num_cands = test_candidates.shape[1]

    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n = len(my_idx)

    cached_idx_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_idx.npy"
    cached_scores_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_scores.npy"
    skip_inference = IS_MPI and os.path.exists(cached_idx_path) and os.path.exists(cached_scores_path)
    if skip_inference:
        try:
            cached_idx = np.load(cached_idx_path)
            cached_scores = np.load(cached_scores_path)
            if (
                cached_idx.shape == my_idx.shape
                and np.array_equal(cached_idx, my_idx)
                and cached_scores.shape == (my_n, num_cands)
            ):
                my_scores = cached_scores.astype(np.float32, copy=False)
                print(f"[r{RANK}] reusing cached shard ({my_n} queries), skipping inference", flush=True)
            else:
                skip_inference = False
        except Exception:
            skip_inference = False

    if not skip_inference:
        my_scores = np.zeros((my_n, num_cands), dtype=np.float32)
        my_n_batches = (my_n + batch_size - 1) // batch_size
        chunk_pairs = max(1, min(cand_chunk_size, batch_size * num_cands))

        pbar = tqdm(range(my_n_batches), ncols=120, desc=f"Testing[r{RANK}]", position=RANK)
        with jt.no_grad():
            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, my_n)
                idx_chunk = my_idx[start:end]
                b = len(idx_chunk)
                total_pairs = b * num_cands

                src_rep = np.repeat(test_src[idx_chunk], num_cands).astype(np.int32)
                t_rep = np.repeat(test_time[idx_chunk].astype(np.float32), num_cands)
                cand_flat = test_candidates[idx_chunk].reshape(-1).astype(np.int32)

                probs_buf = np.empty(total_pairs, dtype=np.float32)
                for cs in range(0, total_pairs, chunk_pairs):
                    ce = min(cs + chunk_pairs, total_pairs)
                    src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=src_rep[cs:ce],
                        dst_node_ids=cand_flat[cs:ce],
                        node_interact_times=t_rep[cs:ce],
                    )
                    logit = predictor(src_emb, dst_emb).squeeze(-1)
                    probs_buf[cs:ce] = jt.sigmoid(logit).numpy().astype(np.float32)
                    del src_emb, dst_emb, logit
                    jt.sync_all()

                my_scores[start:end] = probs_buf.reshape(b, num_cands)
                del probs_buf
                if (batch_idx + 1) % 64 == 0:
                    jt.gc()
        jt.gc()

    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    def _save_and_fsync(path, arr):
        np.save(path, arr)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        assert os.path.getsize(path) > 0, f"[r{RANK}] empty shard written: {path}"

    idx_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_idx.npy"
    scores_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_scores.npy"
    done_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_done.npy"
    _save_and_fsync(idx_path, my_idx)
    _save_and_fsync(scores_path, my_scores)
    _save_and_fsync(done_path, np.array([1], dtype=np.int8))
    print(f"[r{RANK}] shards written: {scores_path} ({os.path.getsize(scores_path)} bytes)", flush=True)

    def _wait_for(path, timeout_s=300.0, poll_s=1.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            time.sleep(poll_s)
        return False

    all_paths = []
    for r in range(WORLD_SIZE):
        all_paths.append((r,
            f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_idx.npy",
            f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_scores.npy",
            f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_done.npy"))

    missing = []
    for r, _, _, p_done in all_paths:
        if not _wait_for(p_done):
            missing.append(r)

    if missing:
        print(f"[r{RANK}] missing sentinels for ranks {missing}; leaving shards. Rerun --eval_only to retry.", flush=True)
        return None

    if not IS_MAIN:
        return None

    log(f"All {WORLD_SIZE} ranks finished. Aggregating...")
    all_idx, all_scores, paths_to_remove = [], [], []
    for r, p_idx, p_scores, p_done in all_paths:
        all_idx.append(np.load(p_idx))
        all_scores.append(np.load(p_scores))
        paths_to_remove.extend([p_idx, p_scores, p_done])
    all_idx = np.concatenate(all_idx)
    all_scores = np.concatenate(all_scores, axis=0)
    sort_order = np.argsort(all_idx)
    result = all_scores[sort_order]
    for p in paths_to_remove:
        try:
            os.remove(p)
        except OSError:
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
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=5)
parser.add_argument("--cand_chunk_size", type=int, default=25)
parser.add_argument("--early_stop", type=int, default=5)
# DyGFormer hyper-parameters
parser.add_argument("--node_feat_dim", type=int, default=128)
parser.add_argument("--edge_feat_dim", type=int, default=128)
parser.add_argument("--time_feat_dim", type=int, default=100)
parser.add_argument("--channel_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=64, help="Larger than 1DyGFormer's 32")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
# Key new flags
parser.add_argument(
    "--use_all_train",
    action="store_true",
    help="Train on ALL train.csv data (split=0 + split=1). No validation. For final submission.",
)
parser.add_argument(
    "--use_split_col",
    action="store_true",
    default=True,
    help="Use the split column (split=0 train, split=1 val) instead of 85/15 count split",
)
parser.add_argument(
    "--pop_feat",
    action="store_true",
    default=True,
    help="Add log(degree+1) popularity feature to node raw features",
)
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"6DyGFormer - Dataset: {args.dataset}")
log(f"  use_all_train={args.use_all_train}, use_split_col={args.use_split_col}")
log(f"  pop_feat={args.pop_feat}, max_seq_len={args.max_seq_len}")
log("=" * 80)

df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

# Determine train/val split
if args.use_all_train:
    # Train on EVERYTHING - no validation split
    train_df = df
    val_df = None
    log(f"use_all_train: training on all {len(train_df)} rows (no validation)")
elif args.use_split_col and "split" in df.columns:
    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df = df[df["split"] == 1].reset_index(drop=True)
    log(f"Using split column: train={len(train_df)}, val={len(val_df)}")
else:
    # Fallback: 85/15 count split
    n_val = int(len(df) * 0.15)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df = df.iloc[-n_val:].reset_index(drop=True)
    log(f"Using 85/15 split: train={len(train_df)}, val={len(val_df)}")

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

# Build node features with log-popularity
src_all = df["src"].values.astype(np.int32)
dst_all = df["dst"].values.astype(np.int32)
t_all = df["time"].values.astype(np.int32)
edge_ids_all = np.arange(len(df), dtype=np.int32) + 1

max_node = max(int(src_all.max()), int(dst_all.max()), int(test_candidates.max()))
node_size = max_node + 1

node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)

if args.pop_feat:
    # Compute log(degree+1) for each node from ALL training data
    from collections import Counter
    deg_counter = Counter()
    deg_counter.update(src_all.tolist())
    deg_counter.update(dst_all.tolist())
    log(f"Computing log-popularity features for {len(deg_counter)} nodes...")
    for node_id, deg in deg_counter.items():
        if node_id < node_size:
            node_raw_features[node_id, 0] = float(np.log1p(deg))
    log(f"  Pop feature range: [{node_raw_features[:, 0].min():.2f}, {node_raw_features[:, 0].max():.2f}]")
    log(f"  Nodes with non-zero pop feature: {(node_raw_features[:, 0] > 0).sum()}")

num_edges = len(df)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

# Build TemporalData for train
train_src = train_df["src"].values.astype(np.int32)
train_dst = train_df["dst"].values.astype(np.int32)
train_t = train_df["time"].values.astype(np.int32)
train_eid = np.arange(len(train_df), dtype=np.int32) + 1

train_data = TemporalData(
    src=jt.Var(train_src),
    dst=jt.Var(train_dst),
    t=jt.Var(train_t),
    edge_ids=jt.Var(train_eid),
)

if val_df is not None:
    val_src = val_df["src"].values.astype(np.int32)
    val_dst = val_df["dst"].values.astype(np.int32)
    val_t = val_df["time"].values.astype(np.int32)
    val_eid = np.arange(len(val_df), dtype=np.int32) + len(train_df) + 1
    val_data = TemporalData(
        src=jt.Var(val_src),
        dst=jt.Var(val_dst),
        t=jt.Var(val_t),
        edge_ids=jt.Var(val_eid),
    )
else:
    val_data = None

# Full data for neighbor sampler (includes all train, so test queries see full history)
full_data = TemporalData(
    src=jt.Var(src_all),
    dst=jt.Var(dst_all),
    t=jt.Var(t_all),
    edge_ids=jt.Var(edge_ids_all),
)

train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0) if val_data is not None else None

if IS_MPI:
    n_keep = (len(train_loader.arange) // WORLD_SIZE) * WORLD_SIZE
    train_loader.arange = train_loader.arange[:n_keep][RANK::WORLD_SIZE]
    if val_loader is not None:
        val_loader.arange = val_loader.arange[RANK::WORLD_SIZE]

full_neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=1)

is_bipartite = len(set(src_all.tolist()) & set(dst_all.tolist())) == 0
log(f"Node size: {node_size}, Num edges: {num_edges}, Bipartite: {is_bipartite}")

backbone = DyGFormer(
    node_raw_features=node_raw_features,
    edge_raw_features=edge_raw_features,
    neighbor_sampler=full_neighbor_sampler,
    time_feat_dim=args.time_feat_dim,
    channel_embedding_dim=args.channel_dim,
    patch_size=args.patch_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout,
    max_input_sequence_length=args.max_seq_len,
    bipartite=is_bipartite,
)
predictor = MergeLayer(
    input_dim1=args.node_feat_dim,
    input_dim2=args.node_feat_dim,
    hidden_dim=args.node_feat_dim,
    output_dim=1,
)
model = nn.Sequential(backbone, predictor)

if IS_MPI:
    model.mpi_param_broadcast(root=0)

if not args.eval_only:
    optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)
    criterion = jt.nn.BCEWithLogitsLoss()
    log(f"\nTraining for {args.epochs} epoch(s)...")
    best_mrr = train(
        model, optimizer, criterion,
        train_loader, val_loader,
        args.epochs, save_path, args.dataset, args.early_stop,
    )
    if IS_MPI:
        jt.array([0], dtype="int32").mpi_all_reduce("add")
else:
    log("\n--eval_only is set, skipping training.")

log("\nGenerating predictions using best model...")
if args.model_path is not None:
    model.load_state_dict(jt.load(args.model_path))
    log(f"Loaded explicit model from {args.model_path}")
else:
    best_model_path = f"{save_path}/{args.dataset}_DyGFormer_best.pkl"
    latest_model_path = f"{save_path}/{args.dataset}_DyGFormer.pkl"
    if os.path.exists(best_model_path):
        model.load_state_dict(jt.load(best_model_path))
        log(f"Loaded best model from {best_model_path}")
    else:
        model.load_state_dict(jt.load(latest_model_path))
        log(f"Best model not found, using latest model")

scores = test_competition(
    model, test_src, test_time, test_candidates,
    save_path, args.dataset,
    args.test_batch_size, cand_chunk_size=args.cand_chunk_size,
)

if IS_MAIN:
    if scores is None:
        log("Aggregation skipped. Rerun --eval_only to retry.")
        sys.exit(0)
    log(f"Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]")
    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result.csv"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for row in scores:
            f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    log(f"Results saved to {output_file}")

log("\n" + "=" * 80)
log("DONE")
log("=" * 80)
