import os
import os.path as osp
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime

# Add JittorGeometric to path
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)

os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
import types

# Monkey-patch: JittorGeometric's DyGFormer calls jt.nn.utils.rnn.pad_sequence,
# which does not exist in this Jittor version. Inject a compatible implementation
# before importing JittorGeometric modules.


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
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.evaluate.evaluators import MRR_Evaluator
import argparse

jt.flags.use_cuda = 1

# MPI / multi-GPU helpers
IS_MPI = jt.in_mpi
RANK = jt.rank if IS_MPI else 0
WORLD_SIZE = jt.world_size if IS_MPI else 1
IS_MAIN = RANK == 0


def log(msg):
    """Only rank 0 prints."""
    if IS_MAIN:
        print(msg)


def sync_model_params(model):
    """All-reduce model parameters across MPI ranks (mean)."""
    if not IS_MPI:
        return
    for p in model.parameters():
        p.update(p.mpi_all_reduce("mean"))


def mpi_barrier():
    """Lightweight MPI barrier via all-reduce."""
    if IS_MPI:
        jt.array([0], dtype="int32").mpi_all_reduce("add")


# =============================================================================
# 0. DyGFormer with Recency Features
# =============================================================================


class DyGFormerRecency(DyGFormer):
    """
    DyGFormer variant that appends 2-dimensional time-decay (recency) features
    to edge raw features during sequence construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_feat_dim += 2
        self.projection_layer_edge = nn.Linear(
            in_features=self.patch_size * self.edge_feat_dim,
            out_features=self.channel_embedding_dim,
            bias=True,
        )

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray, padded_nodes_neighbor_times: np.ndarray,
                     time_encoder):
        node_feats, edge_feats, time_feats = super().get_features(
            node_interact_times, padded_nodes_neighbor_ids, padded_nodes_edge_ids,
            padded_nodes_neighbor_times, time_encoder)

        delta_t = np.maximum(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times, 1e-6)
        tau = 86400 * 30
        recency = np.stack([
            np.exp(-delta_t / tau),
            1.0 / (1.0 + np.log1p(delta_t / 3600.0)),
        ], axis=-1).astype(np.float32)

        mask = (padded_nodes_neighbor_ids != 0).astype(np.float32)[:, :, np.newaxis]
        recency = recency * mask

        recency_var = jt.array(recency)
        edge_feats = jt.concat([edge_feats, recency_var], dim=-1)

        return node_feats, edge_feats, time_feats


# =============================================================================
# 1. Multi-Source Hard Negative Sampler (100-way)
# =============================================================================


class MultiSourceHardNegativeSampler:
    """
    Sample hard negatives from 4 sources for 100-way MRR training:
      1) 33 historical neighbors of src (excluding the positive dst),
         sorted by interaction frequency (highest first).
      2) 17 two-hop neighbors of src (friend-of-friend).
      3) 16 popular nodes sampled from the top-100 most connected dst nodes.
      4) 33 random nodes.
    Deficit from any source is filled with random nodes so that every positive
    gets exactly 99 negatives (100-way total).
    """

    def __init__(self, df, is_bipartite, num_neg=99, warmup_epochs=2, seed=None):
        self.num_neg = num_neg
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.is_bipartite = is_bipartite
        self.min_dst = int(df.dst.min())
        self.max_dst = int(df.dst.max())
        self._rng = np.random.RandomState(seed)

        src_vals = df.src.values.astype(np.int32)
        dst_vals = df.dst.values.astype(np.int32)

        self.src_neighbors = defaultdict(set)
        self.src_neighbor_counts = Counter()
        for s, d in zip(src_vals, dst_vals):
            self.src_neighbors[s].add(d)
            self.src_neighbor_counts[(s, d)] += 1

        self.dst_degrees = Counter()
        for d in dst_vals:
            self.dst_degrees[d] += 1

        top100 = self.dst_degrees.most_common(100)
        self.top100_popular = np.array([node for node, _ in top100], dtype=np.int32)

        self.two_hop = defaultdict(set)
        if is_bipartite:
            item_to_users = defaultdict(set)
            for s, d in zip(src_vals, dst_vals):
                item_to_users[d].add(s)
            for s, items in self.src_neighbors.items():
                for item in items:
                    for other_user in item_to_users.get(item, set()):
                        if other_user != s:
                            self.two_hop[s].update(self.src_neighbors.get(other_user, set()))
        else:
            for s, neighbors in self.src_neighbors.items():
                for n in neighbors:
                    self.two_hop[s].update(self.src_neighbors.get(n, set()))

        for s in self.two_hop:
            self.two_hop[s] -= self.src_neighbors.get(s, set())
            self.two_hop[s].discard(s)

    def step_epoch(self):
        self.epoch += 1

    def _sample_one(self, s, d):
        used = {int(d)}
        candidates = []

        hist = self.src_neighbors.get(s, set()) - used
        if hist:
            hist_sorted = sorted(hist, key=lambda x: self.src_neighbor_counts[(s, x)], reverse=True)
            n_hist = min(33, len(hist_sorted))
            hist_sampled = hist_sorted[:n_hist]
            candidates.extend(hist_sampled)
            used.update(hist_sampled)

        hop2 = self.two_hop.get(s, set()) - used
        n_hop2 = min(17, len(hop2))
        if n_hop2 > 0:
            hop2_sampled = self._rng.choice(list(hop2), size=n_hop2, replace=False).tolist()
            candidates.extend(hop2_sampled)
            used.update(hop2_sampled)

        if len(self.top100_popular) > 0:
            shuffled = self._rng.permutation(self.top100_popular)
            pop_sampled = []
            for node in shuffled:
                if int(node) not in used:
                    pop_sampled.append(int(node))
                if len(pop_sampled) >= 16:
                    break
            candidates.extend(pop_sampled)
            used.update(pop_sampled)

        tries = 0
        max_tries = self.num_neg * 20
        while len(candidates) < self.num_neg and tries < max_tries:
            c = self._rng.randint(self.min_dst, self.max_dst + 1)
            tries += 1
            if c not in used:
                candidates.append(c)
                used.add(c)
        while len(candidates) < self.num_neg:
            candidates.append(self._rng.randint(self.min_dst, self.max_dst + 1))

        return candidates[:self.num_neg]

    def sample(self, src, dst, K=None):
        if K is None:
            K = self.num_neg
        B = len(src)
        if self.epoch < self.warmup_epochs:
            return self._rng.randint(self.min_dst, self.max_dst + 1, size=(B * K,)).astype(np.int32)
        negs = []
        for s, d in zip(src, dst):
            negs.extend(self._sample_one(s, d))
        return np.array(negs, dtype=np.int32)


# =============================================================================
# 2. Ranking Loss
# =============================================================================


def compute_ranking_loss(pos_logit, neg_logit, temperature=1.0):
    B = pos_logit.shape[0]
    K = neg_logit.shape[0] // B
    pos = pos_logit.unsqueeze(1)
    neg = neg_logit.reshape(B, K)
    all_logits = jt.concat([pos, neg], dim=1) / temperature
    log_probs = jt.nn.log_softmax(all_logits, dim=1)
    loss = -log_probs[:, 0].mean()
    return loss


# =============================================================================
# 3. Validation (100-way MRR)
# =============================================================================


def test_val(model, loader, hard_sampler, val_num_neg=99):
    model.eval()
    backbone, predictor = model[0], model[1]
    mrr_eval = MRR_Evaluator()

    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count = 0.0, 0
    loss_sum = 0.0

    loader_tqdm = tqdm(loader, ncols=120, desc="Validation", disable=not IS_MAIN)
    for _, batch_data in enumerate(loader_tqdm):
        src = np.array(batch_data.src).astype(np.int32)
        dst = np.array(batch_data.dst).astype(np.int32)
        t = np.array(batch_data.t).astype(np.float32)

        src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=dst, node_interact_times=t
        )
        pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)
        pos_score = jt.sigmoid(pos_logit).numpy()

        neg_dst = hard_sampler.sample(src, dst, K=val_num_neg)
        src_rep = np.repeat(src, val_num_neg)
        t_rep = np.repeat(t, val_num_neg)
        src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
        )
        neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)
        neg_score = jt.sigmoid(neg_logit).numpy()

        with jt.no_grad():
            loss_sum += float(compute_ranking_loss(pos_logit, neg_logit).item())

        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_score = np.concatenate([pos_score, neg_score])
        ap_sum += float(average_precision_score(y_true, y_score))
        auc_sum += float(roc_auc_score(y_true, y_score))
        n_batches += 1

        batch_mrr = mrr_eval.eval(pos_score, neg_score.reshape(len(src), val_num_neg))
        mrr_sum += float(np.sum(batch_mrr))
        mrr_count += len(batch_mrr)

        del src_emb_pos, dst_emb_pos, pos_logit, pos_score
        del src_emb_neg, neg_dst_emb, neg_logit, neg_score
        jt.sync_all()

    if IS_MPI:
        agg = jt.array(
            [ap_sum, auc_sum, mrr_sum, float(n_batches), float(mrr_count), loss_sum],
            dtype="float32",
        ).mpi_all_reduce("add")
        ap_sum, auc_sum, mrr_sum, n_batches, mrr_count, loss_sum = agg.numpy().tolist()

    if n_batches == 0 or mrr_count == 0:
        return {"AP": 0.0, "AUC": 0.0, "MRR": 0.0, "Loss": 0.0}
    return {
        "AP": ap_sum / n_batches,
        "AUC": auc_sum / n_batches,
        "MRR": mrr_sum / mrr_count,
        "Loss": loss_sum / n_batches,
    }


# =============================================================================
# 4. Training
# =============================================================================


def train(
    model,
    optimizer,
    hard_sampler,
    train_loader,
    val_loader,
    num_epochs,
    save_path,
    dataset_name,
    early_stop_patience=10,
    num_neg=99,
    ranking_temperature=1.0,
    val_num_neg=99,
    grad_clip=1.0,
):
    best_ap = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        hard_sampler.step_epoch()
        model.train()
        backbone, predictor = model[0], model[1]
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch + 1}", disable=not IS_MAIN)

        for batch_idx, batch_data in enumerate(train_tqdm):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)

            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos)

            neg_dst = hard_sampler.sample(src, dst, K=num_neg)
            src_rep = np.repeat(src, num_neg)
            t_rep = np.repeat(t, num_neg)
            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
            )
            neg_logit = predictor(src_emb_neg, neg_dst_emb)

            loss = compute_ranking_loss(pos_logit.squeeze(-1), neg_logit.squeeze(-1), temperature=ranking_temperature)

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()

            sync_model_params(model)
            jt.sync_all()

            train_losses.append(loss.item())
            train_tqdm.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

        local_loss_sum = float(np.sum(train_losses)) if train_losses else 0.0
        local_loss_count = float(len(train_losses))
        if IS_MPI:
            agg_loss = jt.array([local_loss_sum, local_loss_count], dtype="float32").mpi_all_reduce("add")
            global_loss_sum, global_loss_count = agg_loss.numpy().tolist()
            avg_train_loss = global_loss_sum / global_loss_count if global_loss_count > 0 else 0.0
        else:
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0

        log(f"Epoch {epoch + 1}, Train Loss (global avg): {avg_train_loss:.4f}")
        val_res = test_val(model, val_loader, hard_sampler, val_num_neg=val_num_neg)
        log(f"Epoch {epoch + 1}, Val: {val_res}")

        current_ap = val_res["AP"]
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                os.sync()
            mpi_barrier()
            log(f"  -> New best AP: {best_ap:.6f}, model saved!")
        else:
            patience_counter += 1
            log(f"  -> No improvement for {patience_counter} epoch(s), best AP: {best_ap:.6f}")

        if IS_MAIN:
            jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
            os.sync()
        mpi_barrier()

        if patience_counter >= early_stop_patience:
            log(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            log(f"Best validation AP: {best_ap:.6f}")
            break

    return best_ap


# =============================================================================
# 5. Test Inference (100-way MRR evaluation)
# =============================================================================


def test_competition(model, test_src, test_time, test_candidates, tmp_dir, dataset_name, batch_size=50):
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

        pbar = tqdm(range(my_n_batches), ncols=120, desc=f"Testing[r{RANK}]", position=RANK)

        with jt.no_grad():
            for batch_idx in pbar:
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, my_n)
                idx_chunk = my_idx[start:end]
                b = len(idx_chunk)

                src_rep = np.repeat(test_src[idx_chunk], num_cands).astype(np.int32)
                t_rep = np.repeat(test_time[idx_chunk].astype(np.float32), num_cands)
                cand_flat = test_candidates[idx_chunk].reshape(-1).astype(np.int32)

                src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_rep, dst_node_ids=cand_flat, node_interact_times=t_rep
                )
                logit = predictor(src_emb, dst_emb).squeeze(-1)
                probs = jt.sigmoid(logit).numpy().reshape(b, num_cands).astype(np.float32)
                my_scores[start:end] = probs

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
        p_idx = f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_idx.npy"
        p_scores = f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_scores.npy"
        p_done = f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_done.npy"
        all_paths.append((r, p_idx, p_scores, p_done))

    missing = []
    for r, _, _, p_done in all_paths:
        if not _wait_for(p_done):
            missing.append(r)

    if missing:
        print(
            f"[r{RANK}] missing sentinels for ranks {missing}; leaving all shards in place. "
            "Re-run --eval_only to retry.",
            flush=True,
        )
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


# =============================================================================
# 6. Main
# =============================================================================

DATE_STR = datetime.now().strftime("%m%d")
ALGO_NAME = "DyGFormer_recency_hardneg"
DEFAULT_SAVE_DIR = f"./saved_models/{ALGO_NAME}_{DATE_STR}"
DEFAULT_OUTPUT_DIR = f"./data/{ALGO_NAME}_{DATE_STR}"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, help="Model save directory")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU for train/val")
parser.add_argument("--test_batch_size", type=int, default=50, help="Test batch size in queries")
parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience")
parser.add_argument("--node_feat_dim", type=int, default=128)
parser.add_argument("--edge_feat_dim", type=int, default=128)
parser.add_argument("--time_feat_dim", type=int, default=100)
parser.add_argument("--channel_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=48)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_neg", type=int, default=99, help="Number of negatives per positive (99 -> 100-way)")
parser.add_argument("--val_num_neg", type=int, default=99, help="Number of negatives for validation (100-way)")
parser.add_argument("--hard_neg_warmup", type=int, default=2, help="Epochs of uniform warmup before hard negatives")
parser.add_argument("--ranking_temperature", type=float, default=1.0)
parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate saved model")
parser.add_argument("--model_path", type=str, default=None, help="Explicit checkpoint path for eval_only")
args = parser.parse_args()

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

effective_lr = args.lr * WORLD_SIZE if IS_MPI else args.lr
global_batch_size = args.batch_size * WORLD_SIZE

log("=" * 80)
log(f"DyGFormer + Recency + Hard Negatives (dataset1) — {DATE_STR}")
log(f"Dataset: {args.dataset}")
log(f"MPI: in_mpi={IS_MPI}, rank={RANK}, world_size={WORLD_SIZE}")
log(f"Per-GPU batch: {args.batch_size}, Global batch: {global_batch_size}")
log(f"Base LR: {args.lr}, Effective LR (scaled): {effective_lr}")
log(f"Hyper-params: max_seq_len={args.max_seq_len}, patch_size={args.patch_size}, "
    f"num_layers={args.num_layers}, num_heads={args.num_heads}, dropout={args.dropout}, "
    f"channel_dim={args.channel_dim}, time_feat_dim={args.time_feat_dim}")
log(f"Base edge feat dim: {args.edge_feat_dim} (runtime +2 recency dims)")
log(f"100-way MRR: num_neg={args.num_neg}, val_num_neg={args.val_num_neg}")
log("=" * 80)

# Load data
df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

src_np = df["src"].values.astype(np.int32)
dst_np = df["dst"].values.astype(np.int32)
t_np = df["time"].values.astype(np.int32)
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

log(f"Train+Val: {len(df)}, Test: {len(test_df)}")

is_bipartite = len(set(src_np.tolist()) & set(dst_np.tolist())) == 0
log(f"Bipartite: {is_bipartite}")

hard_sampler = MultiSourceHardNegativeSampler(
    df=df,
    is_bipartite=is_bipartite,
    num_neg=args.num_neg,
    warmup_epochs=args.hard_neg_warmup,
    seed=42,
)

# Split 85/15
num_total = len(df)
num_val = int(num_total * 0.15)
num_train = num_total - num_val

train_data = TemporalData(
    src=jt.Var(src_np[:num_train]),
    dst=jt.Var(dst_np[:num_train]),
    t=jt.Var(t_np[:num_train]),
    edge_ids=jt.Var(edge_ids_np[:num_train]),
)
val_data = TemporalData(
    src=jt.Var(src_np[num_train:]),
    dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]),
    edge_ids=jt.Var(edge_ids_np[num_train:]),
)
full_data = TemporalData(
    src=jt.Var(src_np),
    dst=jt.Var(dst_np),
    t=jt.Var(t_np),
    edge_ids=jt.Var(edge_ids_np),
)

train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=None)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=None)

if IS_MPI:
    n_keep = (len(train_loader.arange) // WORLD_SIZE) * WORLD_SIZE
    train_loader.arange = train_loader.arange[:n_keep][RANK::WORLD_SIZE]
    val_loader.arange = val_loader.arange[RANK::WORLD_SIZE]
    log(
        f"[MPI] Train batches/rank: {len(train_loader.arange)}, "
        f"Val batches/rank: {len(val_loader.arange)} "
        f"(global per-step batch = {args.batch_size * WORLD_SIZE})"
    )

full_neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=1)

max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
num_edges = len(df)
log(f"Node size: {node_size}, Num edges: {num_edges}")

node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

backbone = DyGFormerRecency(
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

optimizer = jt.nn.Adam(list(model.parameters()), lr=effective_lr)

if not args.eval_only:
    log(f"\nTraining for {args.epochs} epoch(s) with early stopping (patience={args.early_stop})...")
    best_ap = train(
        model,
        optimizer,
        hard_sampler,
        train_loader,
        val_loader,
        args.epochs,
        save_path,
        args.dataset,
        args.early_stop,
        num_neg=args.num_neg,
        ranking_temperature=args.ranking_temperature,
        val_num_neg=args.val_num_neg,
    )
    mpi_barrier()
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

scores = test_competition(model, test_src, test_time, test_candidates, save_path, args.dataset, args.test_batch_size)

if IS_MAIN:
    if scores is None:
        log("Aggregation skipped. Re-run --eval_only to retry.")
        sys.exit(0)
    log(f"Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]")
    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result.csv"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for row in scores:
            f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    log(f"Result saved to {output_file}")

log("\n" + "=" * 80)
log("DONE")
log("=" * 80)
