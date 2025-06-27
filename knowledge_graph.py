from __future__ import absolute_import, division, print_function

from utils import *


class KnowledgeGraph:
    def __init__(self, dataset):
        self.graph = {}
        self.top_matches = None
        self._load_entities(dataset)
        self._load_reviews(dataset)
        self._load_knowledge(dataset)
        self._clean()

    def _load_entities(self, dataset):
        print('Load entities...')
        total_nodes = 0
        for entity in get_entities():
            self.graph[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for entity_id in range(vocab_size):
                self.graph[entity][entity_id] = {relation: [] for relation in get_relations(entity)}
            total_nodes += vocab_size
        print(f'Total {total_nodes} nodes.')

    def _load_reviews(self, dataset, tfidf_threshold=0.1, freq_threshold=5000):
        print('Load reviews...')
        vocab = dataset.word.vocab
        reviews = [review[2] for review in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        word_distribution = dataset.review.word_distrib

        total_edges = 0
        all_removed_words = []
        for review_id, data in enumerate(dataset.review.data):
            user_id, product_id, review = data
            doc_tfidf = review_tfidf[review_id].toarray()[0]
            remaining_words = [
                word_id for word_id in set(review)
                if doc_tfidf[word_id] >= tfidf_threshold and word_distribution[word_id] <= freq_threshold
            ]
            removed_words = set(review).difference(remaining_words)
            removed_words = [vocab[word_id] for word_id in removed_words]
            all_removed_words.append(removed_words)
            if not remaining_words:
                continue

            self._add_edge(USER, user_id, PURCHASE, PRODUCT, product_id)
            total_edges += 2
            for word_id in remaining_words:
                self._add_edge(USER, user_id, MENTION, WORD, word_id)
                self._add_edge(PRODUCT, product_id, DESCRIBED_AS, WORD, word_id)
                total_edges += 4
        print(f'Total {total_edges} review edges.')

        with open('./tmp/review_removed_words.txt', 'w') as file:
            file.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def _load_knowledge(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print(f'Load knowledge {relation}...')
            data = getattr(dataset, relation).data
            total_edges = 0
            for product_id, entity_ids in enumerate(data):
                if not entity_ids:
                    continue
                for entity_id in set(entity_ids):
                    entity_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, product_id, relation, entity_type, entity_id)
                    total_edges += 2
            print(f'Total {total_edges} {relation} edges.')

    def _add_edge(self, entity_type1, entity_id1, relation, entity_type2, entity_id2):
        self.graph[entity_type1][entity_id1][relation].append(entity_id2)
        self.graph[entity_type2][entity_id2][relation].append(entity_id1)

    def _clean(self):
        print('Remove duplicates...')
        for entity_type in self.graph:
            for entity_id in self.graph[entity_type]:
                for relation in self.graph[entity_type][entity_id]:
                    data = self.graph[entity_type][entity_id][relation]
                    self.graph[entity_type][entity_id][relation] = tuple(sorted(set(data)))

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for entity_type in self.graph:
            self.degrees[entity_type] = {}
            for entity_id in self.graph[entity_type]:
                count = sum(len(self.graph[entity_type][entity_id][relation]) for relation in self.graph[entity_type][entity_id])
                self.degrees[entity_type][entity_id] = count

    def get(self, entity_type, entity_id=None, relation=None):
        data = self.graph
        if entity_type is not None:
            data = data[entity_type]
        if entity_id is not None:
            data = data[entity_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, entity_type, entity_id=None, relation=None):
        return self.get(entity_type, entity_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.graph[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.graph[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        if len(tail_ids) > len(top_match_set):
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {entity: {} for entity in self.graph}
        for entity in self.graph:
            for entity_id in self.graph[entity]:
                for relation in self.graph[entity][entity_id]:
                    if relation not in degrees[entity]:
                        degrees[entity][relation] = []
                    degrees[entity][relation].append(len(self.graph[entity][entity_id][relation]))

        for entity in degrees:
            for relation in degrees[entity]:
                sorted_degrees = sorted(degrees[entity][relation], reverse=True)
                print(entity, relation, sorted_degrees[:10])

    def set_top_matches(self, user_user_match, user_product_match, user_word_match):
        self.top_matches = {
            USER: user_user_match,
            PRODUCT: user_product_match,
            WORD: user_word_match,
        }

    def heuristic_search(self, user_id, product_id, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        get_tails = self.get_tails_given_user if trim_edges else self.get_tails

        pattern = PATH_PATTERN[pattern_id]
        paths = []
        if pattern_id == 1:
            user_words = set(get_tails(USER, user_id, MENTION))
            product_words = set(get_tails(PRODUCT, product_id, DESCRIBED_AS))
            intersect_nodes = user_words.intersection(product_words)
            paths = [(user_id, word, product_id) for word in intersect_nodes]
        elif pattern_id in [11, 12, 13, 14, 15, 16, 17]:
            user_products = set(get_tails(USER, user_id, PURCHASE)).difference([product_id])
            product_nodes = set(get_tails(PRODUCT, product_id, pattern[3][0]))
            if pattern[2][1] == USER:
                product_nodes.difference([user_id])
            for user_product in user_products:
                entity_tail_ids = set(get_tails(PRODUCT, user_product, pattern[2][0]))
                intersect_nodes = entity_tail_ids.intersection(product_nodes)
                paths.extend([(user_id, user_product, node, product_id) for node in intersect_nodes])
        elif pattern_id == 18:
            user_words = set(get_tails(USER, user_id, MENTION))
            product_users = set(get_tails(PRODUCT, product_id, PURCHASE)).difference([user_id])
            for product_user in product_users:
                product_user_words = set(get_tails(USER, product_user, MENTION))
                intersect_nodes = user_words.intersection(product_user_words)
                paths.extend([(user_id, word, product_user, product_id) for word in intersect_nodes])

        return paths


class FoodKnowledgeGraph(KnowledgeGraph):
    def __init__(self, dataset):
        super().__init__(dataset)

    def _load_entities(self, dataset):
        print('Load entities...')
        total_nodes = 0
        for entity in get_food_entities():
            self.graph[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for entity_id in range(vocab_size):
                self.graph[entity][entity_id] = {relation: [] for relation in get_food_relations(entity)}
            total_nodes += vocab_size
        print(f'Total {total_nodes} nodes.')

    def _load_knowledge(self, dataset):
        for relation in [BELONG_TO, HAS_INGREDIENT]:
            print(f'Load knowledge {relation}...')
            data = getattr(dataset, relation).data
            total_edges = 0
            for recipe_id, entity_ids in enumerate(data):
                if not entity_ids:
                    continue
                for entity_id in set(entity_ids):
                    entity_type = get_food_entity_tail(RECIPE, relation)
                    self._add_edge(RECIPE, recipe_id, relation, entity_type, entity_id)
                    total_edges += 2
            print(f'Total {total_edges} {relation} edges.')

    def _load_reviews(self, dataset, tfidf_threshold=0.1, freq_threshold=5000):
        print('Load reviews...')
        vocab = dataset.word.vocab
        reviews = [review[2] for review in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        word_distribution = dataset.review.word_distrib

        total_edges = 0
        all_removed_words = []
        for review_id, data in enumerate(dataset.review.data):
            user_id, recipe_id, review = data
            doc_tfidf = review_tfidf[review_id].toarray()[0]
            remaining_words = [
                word_id for word_id in set(review)
                if doc_tfidf[word_id] >= tfidf_threshold and word_distribution[word_id] <= freq_threshold
            ]
            removed_words = set(review).difference(remaining_words)
            removed_words = [vocab[word_id] for word_id in removed_words]
            all_removed_words.append(removed_words)
            if not remaining_words:
                continue

            self._add_edge(USER, user_id, VIEW, RECIPE, recipe_id)
            total_edges += 2
            for word_id in remaining_words:
                self._add_edge(USER, user_id, MENTION, WORD, word_id)
                self._add_edge(RECIPE, recipe_id, DESCRIBED_AS, WORD, word_id)
                total_edges += 4
        print(f'Total {total_edges} review edges.')

        with open('./tmp/review_removed_words.txt', 'w') as file:
            file.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def get_tails(self, entity_type, entity_id, relation):
        return self.graph[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        tail_type = FOOD_KG_RELATION[entity_type][relation]
        tail_ids = self.graph[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        if len(tail_ids) > len(top_match_set):
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)


def check_test_path(dataset_name, kg):
    test_user_products = load_labels(dataset_name, 'test')
    for user_id in test_user_products:
        for product_id in test_user_products[user_id]:
            count = sum(len(kg.heuristic_search(user_id, product_id, pattern_id)) for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18])
            if count == 0:
                print(user_id, product_id)