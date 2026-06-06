"""
24make_submission.py — submission_honest 的完整实现（一个文件看懂全流程）

原理：对测试集每一行 (src, time, c1..c100)，用若干训练好的 SASRec 模型分别给
      100 个候选打分（sigmoid 概率），加权平均，再行内归一化到 [0,1]，写成 CSV。

前置：每个模型的 test 分数已由 22sasrec_dump.py 生成在
      data/scorers/<dataset>/test_<tag>.npz（形状 [N行, 100]，值=sigmoid概率）。
      （22sasrec_dump.py 做的事：载入 checkpoint → 用用户历史算 user 向量 →
        user·候选item_embedding → sigmoid。就是 SASRec 的打分。）

集成配方（用诚实 100-way 验证集调出来的）：
  dataset1: 17sasrec×1.5 + k31×1.5 + k7×0.6 + seq32×0.4   (val 0.725)
  dataset2: ft2×1.5 + k31×0.6 + seq128k31×1.0             (val 0.421)

用法：
  python 24make_submission.py
"""
import numpy as np
import os

# 每个数据集：{模型tag: 权重}
RECIPE = {
    "dataset1": {"sasrec": 1.5, "k31": 1.5, "k7": 0.6, "seq32": 0.4},
    "dataset2": {"ft2": 1.5, "k31": 0.6, "seq128k31": 1.0},
}


def make(dataset, weights):
    sdir = f"data/scorers/{dataset}"
    # 1) 读各模型的 test 分数 [N,100]，加权求和
    total = None
    for tag, w in weights.items():
        sc = np.load(f"{sdir}/test_{tag}.npz")["score"]  # [N,100] sigmoid概率
        total = w * sc if total is None else total + w * sc

    # 2) 行内 min-max 归一化到 [0,1]（每行 100 个候选相对排序不变）
    mn = total.min(axis=1, keepdims=True)
    mx = total.max(axis=1, keepdims=True)
    norm = (total - mn) / np.where(mx > mn, mx - mn, 1.0)

    # 3) 极小确定性 jitter 打破完全相同分的平局（不影响主排序）
    C = total.shape[1]
    norm = np.clip(norm + ((np.arange(C) % 97) / 97.0 * 1e-6)[None, :], 0.0, 1.0)

    # 4) 写出提交：每行 100 个 8 位小数，候选顺序与 test.csv 一致
    out_dir = f"data/submission_honest/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{dataset}_result.csv"
    with open(out, "w") as f:
        for row in norm:
            f.write(",".join(f"{x:.8f}" for x in row) + "\n")
    print(f"{dataset}: {norm.shape} → {out}  range=[{norm.min():.3f},{norm.max():.3f}]")


if __name__ == "__main__":
    for ds, w in RECIPE.items():
        make(ds, w)
