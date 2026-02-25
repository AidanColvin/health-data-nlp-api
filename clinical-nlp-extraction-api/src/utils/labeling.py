"""
Goal: Build deterministic label mappings.
"""

def build_label_maps(labels):
    uniq = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(uniq)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label
