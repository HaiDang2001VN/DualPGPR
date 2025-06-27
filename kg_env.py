from __future__ import absolute_import, division, print_function
import random
import numpy as np
from utils import *

class KGState:
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise ValueError("history length should be one of {0, 1, 2}")

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed])
        else:
            raise ValueError("history length should be one of {0, 1, 2}")

class BatchKGEnvironment:
    def __init__(self, dataset_str, max_acts, max_path_len=3, state_history=1):
        self.dataset = dataset_str
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        self.kg = load_kg(dataset_str)
        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USER].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self.state_gen = KGState(self.embed_size, history_len=state_history)
        self.state_dim = self.state_gen.dim

        # Compute user-product scores for scaling.
        u_p_scores = np.dot(
            self.embeds[USER] + self.embeds[VIEW][0], self.embeds[RECIPE].T
        ) if self.dataset == FOOD else np.dot(
            self.embeds[USER] + self.embeds[PURCHASE][0], self.embeds[PRODUCT].T
        )
        self.u_p_scales = np.max(u_p_scores, axis=1)

        # Compute path patterns
        self.patterns = []
        for pattern_id in [1, 11, 12, 13, 14, 15] + ([] if self.dataset == FOOD else [16, 17, 18]):
            pattern = PATH_PATTERN[pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]  # pattern contains all relations
            if pattern_id == 1:
                pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False  # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.

    def _has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path):
        return [self._has_pattern(path) for path in batch_path]

    def _get_actions(self, path, done):
        """Compute actions for current node."""
        _, curr_node_type, curr_node_id = path[-1]
        actions = [(SELF_LOOP, curr_node_id)]  # self-loop must be included.

        if done:
            return actions

        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r in relations_nodes:
            next_node_type = FOOD_KG_RELATION[curr_node_type][r] if self.dataset == FOOD else KG_RELATION[curr_node_type][r]
            next_node_ids = [n for n in relations_nodes[r] if (next_node_type, n) not in visited_nodes]
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        if not candidate_acts:
            return actions

        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        user_embed = self.embeds[USER][path[0][-1]]
        scores = []
        if self.dataset == FOOD:
            for r, next_node_id in candidate_acts:
                next_node_type = FOOD_KG_RELATION[curr_node_type][r]
                src_embed = user_embed + self.embeds[VIEW][0] if next_node_type in {RECIPE, WORD} else user_embed + self.embeds[VIEW][0] + self.embeds[r][0]
                score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
                scores.append(score)
        else:
            for r, next_node_id in candidate_acts:
                next_node_type = KG_RELATION[curr_node_type][r]
                src_embed = user_embed + self.embeds[PURCHASE][0] if next_node_type in {PRODUCT, WORD} else user_embed + self.embeds[PURCHASE][0] + self.embeds[r][0]
                score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
                scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    def _get_state(self, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        user_embed = self.embeds[USER][path[0][-1]]
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:  # initial state
            return self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)

        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed, _ = self.embeds[last_relation]  # this can be self-loop!
        if len(path) == 2:
            return self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed, zero_embed)

        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]
        return self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed)

    def _batch_get_state(self, batch_path):
        return np.vstack([self._get_state(path) for path in batch_path])

    def _get_reward(self, path):
        if len(path) <= 2 or not self._has_pattern(path):
            return 0.0

        _, curr_node_type, curr_node_id = path[-1]
        if self.dataset == FOOD and curr_node_type == RECIPE:
            uid = path[0][-1]
            u_vec = self.embeds[USER][uid] + self.embeds[VIEW][0]
            r_vec = self.embeds[RECIPE][curr_node_id]
            score = np.dot(u_vec, r_vec) / self.u_p_scales[uid]
            return max(score, 0.0)
        elif curr_node_type == PRODUCT:
            uid = path[0][-1]
            u_vec = self.embeds[USER][uid] + self.embeds[PURCHASE][0]
            p_vec = self.embeds[PRODUCT][curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
            return max(score, 0.0)

        return 0.0

    def _batch_get_reward(self, batch_path):
        return np.array([self._get_reward(path) for path in batch_path])

    def _is_done(self):
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset(self, uids=None):
        if uids is None:
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]

        self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx):
        assert len(batch_act_idx) == len(self._batch_path)

        for i, act_idx in enumerate(batch_act_idx):
            _, curr_node_type, curr_node_id = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]
            next_node_type = curr_node_type if relation == SELF_LOOP else KG_RELATION[curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self, dropout=0.0):
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = f"Path: {path[0][1]}({path[0][2]})"
            for node in path[1:]:
                msg += f" =={node[0]}=> {node[1]}({node[2]})"
            print(msg)

class BatchCFKGEnvironment(BatchKGEnvironment):
    def __init__(self, dataset_str, max_acts, max_path_len=3, state_history=1):
        super().__init__(dataset_str, max_acts, max_path_len, state_history)

    def _get_actions(self, path, done):
        """Similar to the original method, but return actions with smaller scores."""
        _, curr_node_type, curr_node_id = path[-1]
        actions = [(SELF_LOOP, curr_node_id)]  # self-loop must be included.

        if done:
            return actions

        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r in relations_nodes:
            next_node_type = KG_RELATION[curr_node_type][r]
            next_node_ids = [n for n in relations_nodes[r] if (next_node_type, n) not in visited_nodes]
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        if not candidate_acts:
            return actions

        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        user_embed = self.embeds[USER][path[0][-1]]
        scores = []
        if self.dataset == FOOD:
            for r, next_node_id in candidate_acts:
                next_node_type = FOOD_KG_RELATION[curr_node_type][r]
                src_embed = user_embed + self.embeds[VIEW][0] if next_node_type in {RECIPE, WORD} else user_embed + self.embeds[VIEW][0] + self.embeds[r][0]
                score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
                scores.append(score)
        else:
            for r, next_node_id in candidate_acts:
                next_node_type = KG_RELATION[curr_node_type][r]
                src_embed = user_embed + self.embeds[PURCHASE][0] if next_node_type in {PRODUCT, WORD} else user_embed + self.embeds[PURCHASE][0] + self.embeds[r][0]
                score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
                scores.append(score)
        candidate_idxs = np.argsort(scores)[:self.max_acts]
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]