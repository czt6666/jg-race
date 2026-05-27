import os
import os.path as osp
import sys
import time
from collections import defaultdict, Counter

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


# =============================================================================
# 1. Hard Negative Sampler
# =============================================================================


class HardNegativeSampler:
    """
    Pre-compute structural tables from train edges and sample hard negatives.

    Strategies:
    - hard   : 2-hop neighbors (friend-of-friend for non-bipartite,
                users-who-bought-same-item for bipartite).
    - pop    : degree-weighted random sampling.
    - uniform: plain random in [min_dst, max_dst].
    """

    def __init__(
        self,
        df,
        is_bipartite,
        num_neg=20,
        p_hard=0.5,
        p_pop=0.3,
        p_uniform=0.2,
        warmup_epochs=2,
        seed=None,
    ):
        self.num_neg = num_neg
        self.p_hard = p_hard
        self.p_pop = p_pop
        self.p_uniform = p_uniform
        assert p_hard + p_pop + p_uniform <= 1.0 + 1e-6, "p_hard + p_pop + p_uniform must <= 1"
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.is_bipartite = is_bipartite
        self.min_dst = int(df.dst.min())
        self.max_dst = int(df.dst.max())
        self._rng = np.random.RandomState(seed)

        # Per-src historical neighbors
        self.src_neighbors = defaultdict(set)
        # Per-dst degree (for popularity sampling)
        self.dst_degrees = Counter()

        src_vals = df.src.values
        dst_vals = df.dst.values
        for s, d in zip(src_vals, dst_vals):
            self.src_neighbors[s].add(d)
            self.dst_degrees[d] += 1

        # Popularity distribution
        dsts = np.array(list(self.dst_degrees.keys()), dtype=np.int32)
        degs = np.array([self.dst_degrees[d] for d in dsts], dtype=np.float64)
        self.pop_dsts = dsts
        self.pop_probs = degs / degs.sum()

        # Two-hop tables
        self.two_hop = defaultdict(set)
        if is_bipartite:
            # item -> users who interacted with this item
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

        # Remove direct neighbors and self from 2-hop
        for s in self.two_hop:
            self.two_hop[s] -= self.src_neighbors.get(s, set())
            self.two_hop[s].discard(s)

    def step_epoch(self):
        self.epoch += 1

    def sample(self, src, dst, K=None):
        """
        :param src: ndarray[int32], shape (B,)
        :param dst: ndarray[int32], shape (B,)
        :param K: int, negatives per positive. Defaults to self.num_neg.
        :return: ndarray[int32], shape (B * K,)
        """
        if K is None:
            K = self.num_neg
        B = len(src)

        # Warmup: pure uniform
        if self.epoch < self.warmup_epochs:
            return self._rng.randint(self.min_dst, self.max_dst + 1, size=(B * K,)).astype(np.int32)

        negs = []
        n_hard = int(K * self.p_hard)
        n_pop = int(K * self.p_pop)
        n_uniform = max(0, K - n_hard - n_pop)

        for s, d in zip(src, dst):
            candidates = []
            hist = self.src_neighbors.get(s, set())

            # 1) 2-hop hard negatives
            if n_hard > 0:
                hop2 = list(self.two_hop.get(s, set()))
                if hop2:
                    n = min(len(hop2), n_hard)
                    candidates.extend(self._rng.choice(hop2, size=n, replace=False).tolist())

            # 2) Popularity negatives
            if n_pop > 0 and len(self.pop_dsts) > 0:
                # over-sample then filter collisions
                over = self._rng.choice(self.pop_dsts, size=n_pop * 3, p=self.pop_probs, replace=True)
                for c in over:
                    if c not in hist and c != d and c not in candidates:
                        candidates.append(c)
                    if len(candidates) >= n_hard + n_pop:
                        break

            # 3) Uniform negatives
            if n_uniform > 0:
                tries = 0
                while len(candidates) < K and tries < n_uniform * 10:
                    c = self._rng.randint(self.min_dst, self.max_dst + 1)
                    tries += 1
                    if c not in hist and c != d and c not in candidates:
                        candidates.append(c)

            # Fallback: random fill (may include positives, but extremely rare)
            while len(candidates) < K:
                candidates.append(self._rng.randint(self.min_dst, self.max_dst + 1))

            negs.extend(candidates[:K])

        return np.array(negs, dtype=np.int32)


# =============================================================================
# 2. Ranking Loss
# =============================================================================


def compute_ranking_loss(pos_logit, neg_logit, temperature=1.0):
    """
    Multi-negative softmax ranking loss.

    :param pos_logit: Var, shape (batch_size,)
    :param neg_logit: Var, shape (batch_size * num_neg,)
    :param temperature: float
    :return: scalar Var
    """
    B = pos_logit.shape[0]
    K = neg_logit.shape[0] // B
    pos = pos_logit.unsqueeze(1)  # (B, 1)
    neg = neg_logit.reshape(B, K)  # (B, K)
    all_logits = jt.concat([pos, neg], dim=1) / temperature  # (B, K+1)
    log_probs = jt.nn.log_softmax(all_logits, dim=1)
    loss = -log_probs[:, 0].mean()
    return loss


# =============================================================================
# 3. Validation
# =============================================================================


def test_val(model, loader, hard_sampler, val_num_neg=5):
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

        # Positive pairs
        src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=dst, node_interact_times=t
        )
        pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)
        pos_score = jt.sigmoid(pos_logit).numpy()

        # Hard negatives for validation (fewer than train to save time)
        neg_dst = hard_sampler.sample(src, dst, K=val_num_neg)
        src_rep = np.repeat(src, val_num_neg)
        t_rep = np.repeat(t, val_num_neg)
        src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
        )
        neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)
        neg_score = jt.sigmoid(neg_logit).numpy()

        # Ranking loss (for monitoring only)
        with jt.no_grad():
            loss_sum += float(compute_ranking_loss(pos_logit, neg_logit).item())

        # AP / AUC
        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_score = np.concatenate([pos_score, neg_score])
        ap_sum += float(average_precision_score(y_true, y_score))
        auc_sum += float(roc_auc_score(y_true, y_score))
        n_batches += 1

        # MRR: evaluator expects (B,) pos and (B, K) neg
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
    num_neg=20,
    ranking_temperature=1.0,
    val_num_neg=5,
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

            # Positive pairs
            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos)

            # Hard negatives
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
            train_losses.append(loss.item())
            train_tqdm.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

        log(f"Epoch {epoch + 1}, Train Loss (rank0 shard): {np.mean(train_losses):.4f}")
        val_res = test_val(model, val_loader, hard_sampler, val_num_neg=val_num_neg)
        log(f"Epoch {epoch + 1}, Val: {val_res}")

        current_ap = val_res["AP"]
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                os.sync()
            if IS_MPI:
                jt.array([0], dtype="int32").mpi_all_reduce("add")
            log(f"  -> New best AP: {best_ap:.6f}, model saved!")
        else:
            patience_counter += 1
            log(f"  -> No improvement for {patience_counter} epoch(s), best AP: {best_ap:.6f}")

        if IS_MAIN:
            jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
            os.sync()
        if IS_MPI:
            jt.array([0], dtype="int32").mpi_all_reduce("add")

        if patience_counter >= early_stop_patience:
            log(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            log(f"Best validation AP: {best_ap:.6f}")
            break

    return best_ap


# =============================================================================
# 5. Test Inference (with no_grad + larger default batch)
# =============================================================================


def test_competition(model, test_src, test_time, test_candidates, tmp_dir, dataset_name, batch_size=50):
    model.eval()
    backbone, predictor = model[0], model[1]
    n = len(test_src)
    num_cands = test_candidates.shape[1]

    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n = len(my_idx)

    # Reuse cached shards if present (same logic as 1DyGFormer.py)
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

        # Use no_grad to disable gradient computation and reduce memory
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

                # No explicit del / sync_all here; let GC handle it.
                # numpy() already triggers the needed synchronization.

    # Single-GPU fast path
    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    # Multi-rank: file-based gather (identical to 1DyGFormer.py)
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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Model save directory")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as data_dir)")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size for train/val")
parser.add_argument("--test_batch_size", type=int, default=50, help="Test batch size in queries")
parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience")
# DyGFormer hyper-parameters
parser.add_argument("--node_feat_dim", type=int, default=128)
parser.add_argument("--edge_feat_dim", type=int, default=128)
parser.add_argument("--time_feat_dim", type=int, default=100)
parser.add_argument("--channel_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
# New: hard negative + ranking loss
parser.add_argument("--num_neg", type=int, default=20, help="Number of negatives per positive")
parser.add_argument("--val_num_neg", type=int, default=5, help="Number of negatives for validation")
parser.add_argument("--hard_neg_warmup", type=int, default=2, help="Epochs of uniform warmup before hard negatives")
parser.add_argument("--p_hard", type=float, default=0.5, help="Fraction of hard (2-hop) negatives")
parser.add_argument("--p_pop", type=float, default=0.3, help="Fraction of popularity negatives")
parser.add_argument("--p_uniform", type=float, default=0.2, help="Fraction of uniform negatives")
parser.add_argument("--ranking_temperature", type=float, default=1.0, help="Temperature for ranking softmax")
parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate saved model")
parser.add_argument("--model_path", type=str, default=None, help="Explicit checkpoint path for eval_only")
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"DyGFormer_v2 Competition - Dataset: {args.dataset}")
log(f"MPI: in_mpi={IS_MPI}, rank={RANK}, world_size={WORLD_SIZE}")
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

# Determine bipartite
is_bipartite = len(set(src_np.tolist()) & set(dst_np.tolist())) == 0
log(f"Bipartite: {is_bipartite}")

# Build hard negative sampler (deterministic, every rank can build independently)
hard_sampler = HardNegativeSampler(
    df=df,
    is_bipartite=is_bipartite,
    num_neg=args.num_neg,
    p_hard=args.p_hard,
    p_pop=args.p_pop,
    p_uniform=args.p_uniform,
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

# NOTE: We disable built-in negative sampling; negatives are generated manually per batch.
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=None)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=None)

# Manual data-parallel sharding
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

# Determine sizes
max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
num_edges = len(df)
log(f"Node size: {node_size}, Num edges: {num_edges}")

node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

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

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)

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

log("\n" + "=" * 80)
log("DONE")
log("=" * 80)
