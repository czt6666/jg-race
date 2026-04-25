'''
检查 Cora 赛题数据（data/cora.pkl）的格式与规模：论文数、边、特征、
节点序号是否连续、各划分样本数等。
'''

import os.path as osp
import pickle

import numpy as np


def main():
    data_path = osp.join('data', 'cora.pkl')
    with open(data_path, 'rb') as f:
        raw = pickle.load(f)

    x = np.asarray(raw['x'])
    y = np.asarray(raw['y']).reshape(-1)
    edge_index = np.asarray(raw['edge_index'])
    train_mask = np.asarray(raw['train_mask'], dtype=bool)
    val_mask = np.asarray(raw['val_mask'], dtype=bool)
    test_mask = np.asarray(raw['test_mask'], dtype=bool)

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    num_features = int(raw.get('num_features', x.shape[1]))
    labeled_y = y[y >= 0]
    num_classes = int(
        raw.get(
            'num_classes',
            int(labeled_y.max()) + 1 if labeled_y.size else 0,
        )
    )

    print('=' * 60)
    print('Cora 数据格式检查:', data_path)
    print('=' * 60)

    # 论文（节点）数量
    print('\n【规模】')
    print(f'  论文（节点）数量 num_nodes: {num_nodes}')
    print(f'  边条数 edge_index.shape[1]: {num_edges}（有向边；若图无向通常每条无向边存两条）')
    print(f'  节点特征维度 num_features: {num_features}（x.shape = {x.shape}）')
    print(f'  类别数 num_classes: {num_classes}')

    # 节点序号是否连续：节点用 0..N-1 的行下标表示，边里引用的节点编号应在 [0, N-1]
    print('\n【节点编号 / 序号连续性】')
    print('  节点没有单独「论文 ID」列：第 i 行就是第 i 篇论文（下标 0 .. N-1）。')
    row_min, row_max = 0, num_nodes - 1
    e_min, e_max = int(edge_index.min()), int(edge_index.max())
    print(f'  期望节点下标范围: [{row_min}, {row_max}]')
    print(f'  edge_index 中出现的下标范围: [{e_min}, {e_max}]')
    contiguous = (e_min >= 0 and e_max == num_nodes - 1)
    # 是否每个 0..N-1 都至少出现一次（边或特征行隐含）；Cora 通常全覆盖
    referenced = np.unique(edge_index.flatten())
    all_present = len(referenced) == num_nodes and referenced.min() == 0 and referenced.max() == num_nodes - 1
    print(f'  边中引用的不同节点数: {len(referenced)}（共 N={num_nodes}）')
    print(f'  边下标是否落在 [0, N-1] 且最大为 N-1: {contiguous}')
    print(f'  0..N-1 是否均在边表中出现（通常 Cora 为是）: {all_present}')

    # 划分
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_test = int(test_mask.sum())
    print('\n【训练 / 验证 / 测试划分】')
    print(f'  训练集论文数 (train_mask): {n_train}')
    print(f'  验证集论文数 (val_mask):   {n_val}')
    print(f'  测试集论文数 (test_mask):  {n_test}')
    print(f'  三者之和: {n_train + n_val + n_test}（应等于 num_nodes={num_nodes}）')

    overlap_tv = np.logical_and(train_mask, val_mask).sum()
    overlap_tt = np.logical_and(train_mask, test_mask).sum()
    overlap_vt = np.logical_and(val_mask, test_mask).sum()
    print(f'  train∩val 重叠数: {overlap_tv}（期望 0）')
    print(f'  train∩test 重叠数: {overlap_tt}（期望 0）')
    print(f'  val∩test 重叠数: {overlap_vt}（期望 0）')

    # 标签
    n_hidden = int((y == -1).sum())
    n_labeled = int((y >= 0).sum())
    print('\n【标签 y】')
    print(f'  标签为 -1（隐藏）的节点数: {n_hidden}（通常等于测试集规模）')
    print(f'  标签 >= 0 的节点数: {n_labeled}')
    if n_train > 0:
        yn = y[train_mask]
        print(f'  训练集标签取值范围: [{int(yn.min())}, {int(yn.max())}]')
    if n_val > 0:
        yn = y[val_mask]
        print(f'  验证集标签取值范围: [{int(yn.min())}, {int(yn.max())}]')

    print('\n【原始 pickle 顶层键】')
    print(' ', list(raw.keys()))
    print('=' * 60)


if __name__ == '__main__':
    main()
