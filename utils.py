from __future__ import absolute_import, division, print_function

import os
import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import gzip
import json
import pandas as pd

def parse(path):
    with gzip.open(path, 'rb') as g:
        for line in g:
            yield json.loads(line)

def get_dataframe(path):
    data = {i: d for i, d in enumerate(parse(path))}
    return pd.DataFrame.from_dict(data, orient='index')

def ensure_directory(directory):
    os.makedirs(directory, exist_ok=True)

# Dataset names.
BEAUTY = 'beauty'
CELL = 'cell'
CLOTH = 'cloth'
FOOD = 'food'

# Dataset directories.
DATASET_DIR = {
    BEAUTY: './data/Amazon_Beauty',
    CELL: './data/Amazon_Cellphones',
    CLOTH: './data/Amazon_Clothing',
    FOOD: './data/Food',
}

# Review files.
REVIEW_FILE = {
    CELL: f"{DATASET_DIR[CELL]}/raw/Cell_Phones_and_Accessories.json.gz",
    BEAUTY: f"{DATASET_DIR[BEAUTY]}/raw/Beauty.json.gz",
    CLOTH: f"{DATASET_DIR[CLOTH]}/raw/Clothing_Shoes_and_Jewelry.json.gz",
}

# Metadata files.
META_FILE = {
    CELL: f"{DATASET_DIR[CELL]}/raw/meta_Cell_Phones_and_Accessories.json.gz",
    BEAUTY: f"{DATASET_DIR[BEAUTY]}/raw/meta_Beauty.json.gz",
    CLOTH: f"{DATASET_DIR[CLOTH]}/raw/meta_Clothing_Shoes_and_Jewelry.json.gz",
}

# Model result directories.
TMP_DIR = {
    BEAUTY: './tmp/Amazon_Beauty',
    CELL: './tmp/Amazon_Cellphones',
    CLOTH: './tmp/Amazon_Clothing',
    FOOD: './tmp/Food',
}

# Label files.
LABELS = {
    BEAUTY: (f"{TMP_DIR[BEAUTY]}/train_label.pkl", f"{TMP_DIR[BEAUTY]}/test_label.pkl"),
    CLOTH: (f"{TMP_DIR[CLOTH]}/train_label.pkl", f"{TMP_DIR[CLOTH]}/test_label.pkl"),
    CELL: (f"{TMP_DIR[CELL]}/train_label.pkl", f"{TMP_DIR[CELL]}/test_label.pkl"),
    FOOD: (f"{TMP_DIR[FOOD]}/train_label.pkl", f"{TMP_DIR[FOOD]}/test_label.pkl"),
}

# Entities
USER = 'user'
PRODUCT = 'product'
WORD = 'word'
RPRODUCT = 'related_product'
BRAND = 'brand'
CATEGORY = 'category'

# Relations
PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
PRODUCED_BY = 'produced_by'
BELONG_TO = 'belongs_to'
ALSO_BOUGHT = 'also_bought'
ALSO_VIEWED = 'also_viewed'
BOUGHT_TOGETHER = 'bought_together'
SELF_LOOP = 'self_loop'  # only for kg env

KG_RELATION = {
    USER: {
        PURCHASE: PRODUCT,
        MENTION: WORD,
    },
    WORD: {
        MENTION: USER,
        DESCRIBED_AS: PRODUCT,
    },
    PRODUCT: {
        PURCHASE: USER,
        DESCRIBED_AS: WORD,
        PRODUCED_BY: BRAND,
        BELONG_TO: CATEGORY,
        ALSO_BOUGHT: RPRODUCT,
        ALSO_VIEWED: RPRODUCT,
        BOUGHT_TOGETHER: RPRODUCT,
    },
    BRAND: {
        PRODUCED_BY: PRODUCT,
    },
    CATEGORY: {
        BELONG_TO: PRODUCT,
    },
    RPRODUCT: {
        ALSO_BOUGHT: PRODUCT,
        ALSO_VIEWED: PRODUCT,
        BOUGHT_TOGETHER: PRODUCT,
    }
}

PATH_PATTERN = {
    # length = 3
    1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),
    # length = 4
    11: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
    12: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
    13: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
    14: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
    15: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    16: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    17: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
    18: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
}

relation_name2entity_name = {
    CELL: {
        "category_p_ca": "category",
        "also_buy_related_product_p_re": "related_product",
        "also_buy_product_p_pr": "product",
        "brand_p_br": "brand",
        "also_view_related_product_p_re": "related_product",
        "also_view_product_p_pr": "product",
    }
}

relation2entity = {
    CELL: {
        "category": "category",
        "also_buy_related_product": "related_product",
        "also_buy_product": "product",
        "brand": "brand",
        "also_view_product": "product",
        "also_view_related_product": "related_product",
    }
}

relation_id2plain_name = {
    CELL: {
        "0": "category",
        "1": "also_buy_related_product",
        "2": "related_product",
        "3": "brand",
        "4": "also_view_related_product",
        "5": "related_product"
    }
}

# Food entities
RECIPE = 'recipe'
INGREDIENT = 'ingredient'
TAG = 'tag'

# Food relations
VIEW = 'view'
HAS_INGREDIENT = 'has_ingredient'

# Relation dict
FOOD_KG_RELATION = {
    USER: {
        VIEW: RECIPE,
        MENTION: WORD
    },
    WORD: {
        MENTION: USER,
        DESCRIBED_AS: RECIPE
    },
    RECIPE: {
        VIEW: USER,
        DESCRIBED_AS: WORD,
        BELONG_TO: TAG,
        HAS_INGREDIENT: INGREDIENT
    },
    TAG: {
        BELONG_TO: RECIPE
    },
    INGREDIENT: {
        HAS_INGREDIENT: RECIPE
    }
}

FOOD_PATH_PATTERN = {
    # length = 3
    1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, RECIPE)),
    # length = 4
    11: ((None, USER), (VIEW, RECIPE), (VIEW, USER), (VIEW, RECIPE)),
    12: ((None, USER), (VIEW, RECIPE), (DESCRIBED_AS, WORD), (DESCRIBED_AS, RECIPE)),
    13: ((None, USER), (VIEW, RECIPE), (BELONG_TO, TAG), (BELONG_TO, RECIPE)),
    14: ((None, USER), (VIEW, RECIPE), (HAS_INGREDIENT, INGREDIENT), (HAS_INGREDIENT, RECIPE)),
    15: ((None, USER), (MENTION, WORD), (MENTION, USER), (VIEW, RECIPE)),
}

def get_entities():
    return list(KG_RELATION.keys())

def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())

def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]

def get_food_entities():
    return list(FOOD_KG_RELATION.keys())

def get_food_relations(entity_head):
    return list(FOOD_KG_RELATION[entity_head].keys())

def get_food_entity_tail(entity_head, relation):
    return FOOD_KG_RELATION[entity_head][relation]

def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for doc in docs:
        term_count = {}
        for term_idx in doc:
            term_count[term_idx] = term_count.get(term_idx, 0) + 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    term_freq = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(term_freq)
    return tfidf

def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.handlers.RotatingFileHandler(logname, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_dataset(dataset, dataset_obj):
    dataset_file = f"{TMP_DIR[dataset]}/dataset.pkl"
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)

def load_dataset(dataset):
    dataset_file = f"{TMP_DIR[dataset]}/dataset.pkl"
    with open(dataset_file, 'rb') as f:
        return pickle.load(f)

def save_labels(dataset, labels, mode='train'):
    label_file = LABELS[dataset][0] if mode == 'train' else LABELS[dataset][1]
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)

def load_labels(dataset, mode='train'):
    label_file = LABELS[dataset][0] if mode == 'train' else LABELS[dataset][1]
    with open(label_file, 'rb') as f:
        return pickle.load(f)

def save_embed(dataset, embed):
    embed_file = f"{TMP_DIR[dataset]}/transe_embed.pkl"
    with open(embed_file, 'wb') as f:
        pickle.dump(embed, f)

def load_embed(dataset):
    embed_file = f"{TMP_DIR[dataset]}/transe_embed.pkl"
    print(f'Load embedding: {embed_file}')
    with open(embed_file, 'rb') as f:
        return pickle.load(f)

def save_kg(dataset, kg):
    kg_file = f"{TMP_DIR[dataset]}/kg.pkl"
    with open(kg_file, 'wb') as f:
        pickle.dump(kg, f)

def load_kg(dataset):
    kg_file = f"{TMP_DIR[dataset]}/kg.pkl"
    with open(kg_file, 'rb') as f:
        return pickle.load(f)

def save_pred_paths(folder_path, pred_paths, train_labels):
    print("Normalizing items scores...")
    # Get min and max score to perform normalization between 0 and 1
    score_list = [
        float(path[0])
        for uid, pid_dict in pred_paths.items()
        for pid, path_list in pid_dict.items()
        if pid not in set(train_labels[uid])
        for path in path_list
    ]
    min_score, max_score = min(score_list), max(score_list)

    print("Saving predicted paths...")
    records = [
        [uid, pid, str((float(path[0]) - min_score) / (max_score - min_score)), path[1], ' '.join(map(str, sum(path[2], ())))]
        for uid, pid_dict in pred_paths.items()
        for pid, path_list in pid_dict.items()
        if pid not in set(train_labels[uid])
        for path in path_list
    ]
    with open(f"{folder_path}/pred_paths.pkl", 'wb') as pred_paths_file:
        pickle.dump(records, pred_paths_file)