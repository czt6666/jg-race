'''
将 data/cora.pkl 中的全部内容导出到单个文件。

默认：cora_raw.npz（numpy 压缩包，含所有数组与标量，可用 np.load 读回）
可选：--format txt  单个大文本（含完整 x/y/边/掩码，文件会很大）
'''

import argparse
import json
import os.path as osp
import pickle

import numpy as np


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def export_npz(raw, out_path):
    x = np.asarray(raw['x'])
    y = np.asarray(raw['y']).reshape(-1)
    edge_index = np.asarray(raw['edge_index'])
    train_mask = np.asarray(raw['train_mask'])
    val_mask = np.asarray(raw['val_mask'])
    test_mask = np.asarray(raw['test_mask'])
    num_classes = int(raw.get('num_classes', int(y[y >= 0].max()) + 1 if np.any(y >= 0) else 0))
    num_features = int(raw.get('num_features', x.shape[1]))

    np.savez_compressed(
        out_path,
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=np.array(num_classes),
        num_features=np.array(num_features),
    )
    print(f'已写入 NPZ: {out_path}')
    print(f'  x {x.shape}, y {y.shape}, edge_index {edge_index.shape}')


def export_txt(raw, out_path):
    x = np.asarray(raw['x'])
    y = np.asarray(raw['y']).reshape(-1)
    edge_index = np.asarray(raw['edge_index'])
    train_mask = np.asarray(raw['train_mask'])
    val_mask = np.asarray(raw['val_mask'])
    test_mask = np.asarray(raw['test_mask'])
    num_classes = int(raw.get('num_classes', int(y[y >= 0].max()) + 1 if np.any(y >= 0) else 0))
    num_features = int(raw.get('num_features', x.shape[1]))

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# cora.pkl 全量导出（单文件文本）\n')
        f.write(f'# num_nodes={x.shape[0]} num_edges={edge_index.shape[1]} '
                f'num_features={num_features} num_classes={num_classes}\n\n')

        f.write('## META_JSON（pickle 中除 x/y/edge/mask 外的其余键）\n')
        meta = {}
        skip = {'x', 'y', 'edge_index', 'train_mask', 'val_mask', 'test_mask'}
        for k, v in raw.items():
            if k in skip:
                continue
            if isinstance(v, np.ndarray):
                meta[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                meta[k] = v.item()
            else:
                meta[k] = v
        f.write(json.dumps(meta, ensure_ascii=False))
        f.write('\n\n')

        f.write('## Y (one label per line, node index = line number starting 0)\n')
        for i in range(y.size):
            f.write(f'{int(y[i])}\n')

        f.write('\n## EDGE_INDEX (source\\ttarget per line)\n')
        for e in range(edge_index.shape[1]):
            f.write(f'{int(edge_index[0, e])}\t{int(edge_index[1, e])}\n')

        f.write('\n## MASKS (train val test as 0/1 per line, same node order as Y)\n')
        for i in range(x.shape[0]):
            f.write(f'{int(train_mask[i])}\t{int(val_mask[i])}\t{int(test_mask[i])}\n')

        f.write('\n## X (each row = one node, comma-separated floats)\n')
        for i in range(x.shape[0]):
            f.write(','.join(str(float(v)) for v in x[i]))
            f.write('\n')

    print(f'已写入 TXT: {out_path}（体积较大，请耐心等待）')


def main():
    p = argparse.ArgumentParser(description='导出 cora.pkl 到单个文件')
    p.add_argument('--in_path', default=osp.join('data', 'cora.pkl'), help='输入 pickle')
    p.add_argument('--out_path', default='', help='输出路径（默认由格式决定）')
    p.add_argument('--format', choices=('npz', 'txt'), default='npz', help='输出格式')
    args = p.parse_args()

    if not osp.isfile(args.in_path):
        raise SystemExit(f'找不到输入文件: {args.in_path}')

    raw = load_pkl(args.in_path)

    if args.out_path:
        out = args.out_path
    else:
        base = osp.splitext(osp.basename(args.in_path))[0]
        out = f'{base}_raw.{args.format}'

    if args.format == 'npz':
        export_npz(raw, out)
    else:
        export_txt(raw, out)


if __name__ == '__main__':
    main()
