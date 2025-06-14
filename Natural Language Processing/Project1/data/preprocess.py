import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_ltcr_data_from_csv(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)

def load_pheme(data_dir, test_size=0.2, random_state=42):
    """
    递归加载 PHEME 数据集中的所有 source-tweets JSON 文件，
    并基于同目录下的 annotation.json 生成标签：0=false, 1=true, 2=unverified
    """
    def convert_annotations(annotation):
        # PHEME annotation uses 'is_rumour': 'rumour' or 'nonrumour'
        if annotation == 'rumour':
            return 1
        elif annotation == 'nonrumour':
            return 0
        else:
            return 2

    texts, labels = [], []
    for root, dirs, files in os.walk(data_dir):
        # 仅在名为 source-tweets 的目录下处理文本
        if os.path.basename(root) == 'source-tweets':
            parent = os.path.dirname(root)
            ann_file = os.path.join(parent, 'annotation.json')
            ann_items = {}
            if os.path.isfile(ann_file):
                with open(ann_file, 'r', encoding='utf-8', errors='ignore') as af:
                    try:
                        ann_items = json.load(af)
                    except json.JSONDecodeError:
                        ann_items = {}

            for fname in files:
                if not fname.endswith('.json'):
                    continue
                # 根据 is_rumour 字段决定标签
                lbl = convert_annotations(ann_items['is_rumour'])
                src_path = os.path.join(root, fname)
                with open(src_path, 'r', encoding='utf-8', errors='ignore') as sf:
                    try:
                        data = json.load(sf)
                    except json.JSONDecodeError:
                        continue
                items = data if isinstance(data, list) else [data]
                for item in items:
                    text = item.get('text') or item.get('tweet') or ''
                    texts.append(text)
                    labels.append(lbl)

    if not texts:
        raise ValueError(f"No PHEME source JSON found under {data_dir}")
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)


def load_data(config):
    """
    根据 config['dataset_name'] 选择数据集加载函数。
    支持 'LTCR', 'PHEME'
    """
    name = config.get('dataset_name', 'LTCR').lower()
    if name == 'ltcr':
        return load_ltcr_data_from_csv(config['dataset_path'] + 'LTCR.csv')
    elif name == 'pheme':
        return load_pheme(config['dataset_path'])
    else:
        raise ValueError(f"Unknown dataset: {config['dataset_name']}")
