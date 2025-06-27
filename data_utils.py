from __future__ import absolute_import, division, print_function

import numpy as np
import gzip
from easydict import EasyDict as edict
import random


class AmazonDataset:
    """This class is used to load data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir if data_dir.endswith('/') else f'{data_dir}/'
        self.review_file = f'{set_name}.txt.gz'
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

    def _load_file(self, filename):
        with gzip.open(f'{self.data_dir}{filename}', 'r') as f:
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self):
        """Load 6 global entities from data files."""
        entity_files = edict(
            user='users.txt.gz',
            product='product.txt.gz',
            word='vocab.txt.gz',
            related_product='related_product.txt.gz',
            brand='brand.txt.gz',
            category='category.txt.gz',
        )
        for name, file in entity_files.items():
            vocab = self._load_file(file)
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print(f'Load {name} of size {len(vocab)}')

    def load_reviews(self):
        """Load user-product reviews from train/test data files."""
        review_data = []
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0

        for line in self._load_file(self.review_file):
            user_idx, product_idx, words = line.split('\t')
            word_indices = list(map(int, words.split(' ')))
            review_data.append((int(user_idx), int(product_idx), word_indices))
            product_distrib[int(product_idx)] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)

        self.review = edict(
            data=review_data,
            size=len(review_data),
            product_distrib=product_distrib,
            product_uniform_distrib=np.ones(self.product.vocab_size),
            word_distrib=word_distrib,
            word_count=word_count,
            review_distrib=np.ones(len(review_data))
        )
        print(f'Load review of size {self.review.size}, word count = {word_count}')

    def load_product_relations(self):
        """Load 5 product -> ? relations."""
        product_relations = edict(
            produced_by=('brand_p_b.txt.gz', self.brand),
            belongs_to=('category_p_c.txt.gz', self.category),
            also_bought=('also_bought_p_p.txt.gz', self.related_product),
            also_viewed=('also_viewed_p_p.txt.gz', self.related_product),
            bought_together=('bought_together_p_p.txt.gz', self.related_product),
        )
        for name, (file, entity) in product_relations.items():
            relation = edict(
                data=[],
                et_vocab=entity.vocab,
                et_distrib=np.zeros(entity.vocab_size)
            )
            for line in self._load_file(file):
                knowledge = [int(x) for x in line.split() if x]
                for x in knowledge:
                    relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print(f'Load {name} of size {len(relation.data)}')

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i, freq in enumerate(self.review.word_distrib):
            if freq > 0:
                self.word_sampling_rate[i] = min((np.sqrt(freq / threshold) + 1) * threshold / freq, 1.0)


class AmazonDataLoader:
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.review_size = self.dataset.review.size
        self.product_relations = ['produced_by', 'belongs_to', 'also_bought', 'also_viewed', 'bought_together']
        self.finished_word_num = 0
        self.reset()

    def reset(self):
        self.review_seq = np.random.permutation(self.review_size)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8]."""
        batch = []
        while len(batch) < self.batch_size and self._has_next:
            review_idx = self.review_seq[self.cur_review_i]
            user_idx, product_idx, text_list = self.dataset.review.data[review_idx]
            product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}

            while len(batch) < self.batch_size:
                word_idx = text_list[self.cur_word_i]
                if random.random() < self.dataset.word_sampling_rate[word_idx]:
                    data = [user_idx, product_idx, word_idx] + [
                        random.choice(product_knowledge[pr]) if product_knowledge[pr] else -1
                        for pr in self.product_relations
                    ]
                    batch.append(data)

                self.cur_word_i += 1
                self.finished_word_num += 1
                if self.cur_word_i >= len(text_list):
                    self.cur_review_i += 1
                    if self.cur_review_i >= self.review_size:
                        self._has_next = False
                        break
                    self.cur_word_i = 0
                    break

        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next


class FoodDataset(AmazonDataset):
    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir if data_dir.endswith('/') else f'{data_dir}/'
        self.review_file = f'{set_name}.txt.gz'
        self.load_entities()
        self.load_recipe_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

    def load_entities(self):
        entity_files = edict(
            user='users.txt.gz',
            recipe='recipe.txt.gz',
            word='vocab.txt.gz',
            tag='tag.txt.gz',
            ingredient='ingredient.txt.gz',
        )
        for name, file in entity_files.items():
            vocab = self._load_file(file)
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print(f'Load {name} of size {len(vocab)}')

    def load_reviews(self):
        review_data = []
        recipe_distrib = np.zeros(self.recipe.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0

        for line in self._load_file(self.review_file):
            user_idx, recipe_idx, words = line.split('\t')
            word_indices = list(map(int, words.split(' ')))
            review_data.append((int(user_idx), int(recipe_idx), word_indices))
            recipe_distrib[int(recipe_idx)] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)

        self.review = edict(
            data=review_data,
            size=len(review_data),
            recipe_distrib=recipe_distrib,
            recipe_uniform_distrib=np.ones(self.recipe.vocab_size),
            word_distrib=word_distrib,
            word_count=word_count,
            review_distrib=np.ones(len(review_data))
        )
        print(f'Load review of size {self.review.size}, word count = {word_count}')

    def load_recipe_relations(self):
        recipe_relations = edict(
            belongs_to=('tag_r_t.txt.gz', self.tag),
            has_ingredient=('ingredient_r_i.txt.gz', self.ingredient),
        )
        for name, (file, entity) in recipe_relations.items():
            relation = edict(
                data=[],
                et_vocab=entity.vocab,
                et_distrib=np.zeros(entity.vocab_size)
            )
            for line in self._load_file(file):
                knowledge = [int(x) for x in line.split() if x]
                for x in knowledge:
                    relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print(f'Load {name} of size {len(relation.data)}')


class FoodDataLoader(AmazonDataLoader):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.review_size = self.dataset.review.size
        self.recipe_relations = ['belongs_to', 'has_ingredient']
        self.finished_word_num = 0
        self.reset()

    def get_batch(self):
        """Return a matrix of [batch_size x 5]."""
        batch = []
        while len(batch) < self.batch_size and self._has_next:
            review_idx = self.review_seq[self.cur_review_i]
            user_idx, recipe_idx, text_list = self.dataset.review.data[review_idx]
            recipe_knowledge = {r: getattr(self.dataset, r).data[recipe_idx] for r in self.recipe_relations}

            while len(batch) < self.batch_size:
                word_idx = text_list[self.cur_word_i]
                if random.random() < self.dataset.word_sampling_rate[word_idx]:
                    data = [user_idx, recipe_idx, word_idx] + [
                        random.choice(recipe_knowledge[r]) if recipe_knowledge[r] else -1
                        for r in self.recipe_relations
                    ]
                    batch.append(data)

                self.cur_word_i += 1
                self.finished_word_num += 1
                if self.cur_word_i >= len(text_list):
                    self.cur_review_i += 1
                    if self.cur_review_i >= self.review_size:
                        self._has_next = False
                        break
                    self.cur_word_i = 0
                    break

        return np.array(batch)