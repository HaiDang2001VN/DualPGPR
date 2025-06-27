from __future__ import absolute_import, division, print_function

import os
import gzip
import argparse

from utils import *
from data_utils import AmazonDataset, FoodDataset
from knowledge_graph import KnowledgeGraph, FoodKnowledgeGraph


def generate_labels(dataset_name, mode='train'):
    """
    Generate labels for the given dataset and mode.

    Args:
        dataset_name (str): The name of the dataset.
        mode (str, optional): The mode for which labels are generated. Defaults to 'train'.

    Returns:
        None
    """
    review_file = f'{DATASET_DIR[dataset_name]}/{mode}.txt.gz'
    user_products = {}  # {user_id: [product_id,...], ...}
    with gzip.open(review_file, 'r') as file:
        for line in file:
            line = line.decode('utf-8').strip()
            user_id, product_id = map(int, line.split('\t')[:2])
            user_products.setdefault(user_id, []).append(product_id)
    save_labels(dataset_name, user_products, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {BEAUTY, CELL, CLOTH, FOOD}.')
    args = parser.parse_args()

    dataset_name = args.dataset

    # Create AmazonDataset instance for dataset.
    print(f'Load {dataset_name} dataset from file...')
    os.makedirs(TMP_DIR[dataset_name], exist_ok=True)
    dataset = FoodDataset(DATASET_DIR[dataset_name]) if dataset_name == FOOD else AmazonDataset(DATASET_DIR[dataset_name])
    save_dataset(dataset_name, dataset)

    # Generate knowledge graph instance.
    print(f'Create {dataset_name} knowledge graph from dataset...')
    dataset = load_dataset(dataset_name)
    kg = FoodKnowledgeGraph(dataset) if dataset_name == FOOD else KnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(dataset_name, kg)

    # Generate train/test labels.
    print(f'Generate {dataset_name} train/test labels.')
    generate_labels(dataset_name, 'train')
    generate_labels(dataset_name, 'test')


if __name__ == '__main__':
    main()