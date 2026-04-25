"""
Cora 节点分类：两层层 GCN，训练后导出测试集预测为 JSON。
依赖：jittor、JittorGeometric（本仓库请确保可 import jittor_geometric）。
数据：默认读取脚本同目录下 data/cora.pkl。

Windows 说明：Jittor 若检测到 jtcuda/nvcc，会用 HAS_CUDA 编译 data.cc，易与 MSVC 报 C2440。
在 import jittor 之前将 nvcc_path 置空可强制走 CPU 编译（默认行为，见下方逻辑）。
需要 GPU 时加 --cuda，并自行保证本机 CUDA/MSVC 与 Jittor 版本兼容。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 必须在 import jittor 之前生效（compiler 读取环境变量）
if os.name == "nt" and "--cuda" not in sys.argv and "nvcc_path" not in os.environ:
    os.environ["nvcc_path"] = ""

import jittor as jt
import numpy as np
from jittor import nn

for _root in (_SCRIPT_DIR, os.path.join(_SCRIPT_DIR, "JittorGeometric-main")):
    if os.path.isdir(os.path.join(_root, "jittor_geometric")) and _root not in sys.path:
        sys.path.insert(0, _root)

from jittor_geometric.nn import GCNConv
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.ops import cootocsc, cootocsr


def row_normalize_features(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    return x / s


class GCNNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        spmm: bool,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, spmm=spmm)
        self.conv2 = GCNConv(hidden_channels, out_channels, spmm=spmm)
        self.dropout = dropout

    def execute(self, x, csc, csr):
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)
        return x


def load_cora_pkl(path: str) -> SimpleNamespace:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    x = np.asarray(raw["x"], dtype=np.float32)
    y = np.asarray(raw["y"], dtype=np.int64).reshape(-1)
    edge_index = np.asarray(raw["edge_index"], dtype=np.int32)
    train_mask = np.asarray(raw["train_mask"], dtype=bool)
    val_mask = np.asarray(raw["val_mask"], dtype=bool)
    test_mask = np.asarray(raw["test_mask"], dtype=bool)
    num_classes = int(raw.get("num_classes", int(y.max()) + 1))
    num_features = int(raw.get("num_features", x.shape[1]))
    return SimpleNamespace(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        num_features=num_features,
    )


def build_graph(data: SimpleNamespace):
    v_num = data.x.shape[0]
    x_np = row_normalize_features(data.x)
    x = jt.array(x_np)
    y = jt.array(data.y)
    train_mask = jt.array(data.train_mask)
    val_mask = jt.array(data.val_mask)
    test_mask = jt.array(data.test_mask)

    edge_index = jt.array(data.edge_index)
    edge_weight = None
    edge_index, edge_weight = gcn_norm(
        edge_index, edge_weight, v_num, improved=False, add_self_loops=True
    )
    with jt.no_grad():
        csc = cootocsc(edge_index, edge_weight, v_num)
        csr = cootocsr(edge_index, edge_weight, v_num)

    return SimpleNamespace(
        x=x,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        csc=csc,
        csr=csr,
        v_num=v_num,
    )


def accuracy(logits, y, mask):
    pred, _ = jt.argmax(logits[mask], dim=1)
    correct = pred.equal(y[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "data", "cora.pkl"),
        help="cora.pkl 路径",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "cora_test_pred.json"),
        help="测试集预测输出 JSON",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=50, help="验证集早停")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spmm", action="store_true", help="GCNConv 使用 spmm（GPU 上通常更快）")
    parser.add_argument("--cuda", action="store_true", help="启用 CUDA（需环境支持）")
    args = parser.parse_args()

    jt.misc.set_global_seed(args.seed)
    jt.flags.use_cuda = 1 if args.cuda else 0

    raw = load_cora_pkl(args.data_path)
    graph = build_graph(raw)

    model = GCNNet(
        raw.num_features,
        args.hidden,
        raw.num_classes,
        args.dropout,
        spmm=args.spmm,
    )
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = 0.0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(graph.x, graph.csc, graph.csr)
        loss = nn.cross_entropy_loss(logits[graph.train_mask], graph.y[graph.train_mask])
        optimizer.step(loss)

        model.eval()
        logits = model(graph.x, graph.csc, graph.csr)
        val_acc = accuracy(logits, graph.y, graph.val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 50 == 0 or epoch == 1:
            train_acc = accuracy(logits, graph.y, graph.train_mask)
            print(f"Epoch {epoch:04d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if bad >= args.patience:
            print(f"Early stop at epoch {epoch}, best_val_acc={best_val:.4f}")
            break

    if best_state is not None:
        model.load_parameters(best_state)

    model.eval()
    logits = model(graph.x, graph.csc, graph.csr)
    pred, _ = jt.argmax(logits[graph.test_mask], dim=1)
    pred_np = pred.numpy()
    test_idx = np.nonzero(raw.test_mask)[0]

    out = {int(node_id): int(pred_np[i]) for i, node_id in enumerate(test_idx)}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已写入测试集预测: {args.out_json} （共 {len(out)} 个节点）")


if __name__ == "__main__":
    main()
