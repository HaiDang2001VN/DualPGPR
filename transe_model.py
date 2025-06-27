from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn

from utils import *


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            related_product=edict(vocab_size=dataset.related_product.vocab_size),
            brand=edict(vocab_size=dataset.brand.vocab_size),
            category=edict(vocab_size=dataset.category.vocab_size),
        )
        for entity in self.entities:
            embed = self._create_entity_embedding(self.entities[entity].vocab_size)
            setattr(self, entity, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            purchase=edict(
                et="product",
                et_distrib=self._normalize_distribution(dataset.review.product_uniform_distrib),
            ),
            mentions=edict(
                et="word", 
                et_distrib=self._normalize_distribution(dataset.review.word_distrib)
            ),
            describe_as=edict(
                et="word", 
                et_distrib=self._normalize_distribution(dataset.review.word_distrib)
            ),
            produced_by=edict(
                et="brand",
                et_distrib=self._normalize_distribution(dataset.produced_by.et_distrib),
            ),
            belongs_to=edict(
                et="category",
                et_distrib=self._normalize_distribution(dataset.belongs_to.et_distrib),
            ),
            also_bought=edict(
                et="related_product",
                et_distrib=self._normalize_distribution(dataset.also_bought.et_distrib),
            ),
            also_viewed=edict(
                et="related_product",
                et_distrib=self._normalize_distribution(dataset.also_viewed.et_distrib),
            ),
            bought_together=edict(
                et="related_product",
                et_distrib=self._normalize_distribution(dataset.bought_together.et_distrib),
            ),
        )
        for relation in self.relations:
            embed = self._create_relation_embedding()
            setattr(self, relation, embed)
            bias = self._create_relation_bias(len(self.relations[relation].et_distrib))
            setattr(self, f"{relation}_bias", bias)

    def _create_entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
        Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _create_relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _create_relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _normalize_distribution(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float64), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_indices):
        return self.compute_loss(batch_indices)

    def compute_loss(self, batch_indices):
        """Compute knowledge graph negative sampling loss.
        batch_indices: batch_size * 8 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        user_indices = batch_indices[:, 0]
        product_indices = batch_indices[:, 1]
        word_indices = batch_indices[:, 2]
        brand_indices = batch_indices[:, 3]
        category_indices = batch_indices[:, 4]
        related_product1_indices = batch_indices[:, 5]
        related_product2_indices = batch_indices[:, 6]
        related_product3_indices = batch_indices[:, 7]

        regularizations = []

        # user + purchase -> product
        up_loss, up_embeds = self._negative_sampling_loss(
            "user", "purchase", "product", user_indices, product_indices
        )
        regularizations.extend(up_embeds)
        loss = up_loss

        # user + mentions -> word
        uw_loss, uw_embeds = self._negative_sampling_loss(
            "user", "mentions", "word", user_indices, word_indices
        )
        regularizations.extend(uw_embeds)
        loss += uw_loss

        # product + describe_as -> word
        pw_loss, pw_embeds = self._negative_sampling_loss(
            "product", "describe_as", "word", product_indices, word_indices
        )
        regularizations.extend(pw_embeds)
        loss += pw_loss

        # product + produced_by -> brand
        pb_loss, pb_embeds = self._negative_sampling_loss(
            "product", "produced_by", "brand", product_indices, brand_indices
        )
        if pb_loss is not None:
            regularizations.extend(pb_embeds)
            loss += pb_loss

        # product + belongs_to -> category
        pc_loss, pc_embeds = self._negative_sampling_loss(
            "product", "belongs_to", "category", product_indices, category_indices
        )
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss

        # product + also_bought -> related_product1
        pr1_loss, pr1_embeds = self._negative_sampling_loss(
            "product", "also_bought", "related_product", product_indices, related_product1_indices
        )
        if pr1_loss is not None:
            regularizations.extend(pr1_embeds)
            loss += pr1_loss

        # product + also_viewed -> related_product2
        pr2_loss, pr2_embeds = self._negative_sampling_loss(
            "product", "also_viewed", "related_product", product_indices, related_product2_indices
        )
        if pr2_loss is not None:
            regularizations.extend(pr2_embeds)
            loss += pr2_loss

        # product + bought_together -> related_product3
        pr3_loss, pr3_embeds = self._negative_sampling_loss(
            "product", "bought_together", "related_product", product_indices, related_product3_indices
        )
        if pr3_loss is not None:
            regularizations.extend(pr3_embeds)
            loss += pr3_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = sum(torch.norm(term) for term in regularizations)
            loss += self.l2_lambda * l2_loss

        return loss

    def _negative_sampling_loss(self, entity_head, relation, entity_tail, entity_head_indices, entity_tail_indices):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_indices >= 0
        fixed_entity_head_indices = entity_head_indices[mask]
        fixed_entity_tail_indices = entity_tail_indices[mask]
        if fixed_entity_head_indices.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, f"{relation}_bias")  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss(
            entity_head_embedding,
            entity_tail_embedding,
            fixed_entity_head_indices,
            fixed_entity_tail_indices,
            relation_vec,
            relation_bias_embedding,
            self.num_neg_samples,
            entity_tail_distrib,
        )


def kg_neg_loss(
    entity_head_embed,
    entity_tail_embed,
    entity_head_indices,
    entity_tail_indices,
    relation_vec,
    relation_bias_embed,
    num_samples,
    distrib,
):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_indices:
        entity_tail_indices:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """
    batch_size = entity_head_indices.size(0)
    entity_head_vec = entity_head_embed(entity_head_indices)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_indices)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_indices).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]


class FoodKnowledgeEmbedding(KnowledgeEmbedding):
    def __init__(self, dataset, args):
        nn.Module.__init__(self)
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        
        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            recipe=edict(vocab_size=dataset.recipe.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            tag=edict(vocab_size=dataset.tag.vocab_size),
            ingredient=edict(vocab_size=dataset.ingredient.vocab_size),
        )
        for entity in self.entities:
            embed = self._create_entity_embedding(self.entities[entity].vocab_size)
            setattr(self, entity, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            view=edict(
                et="recipe",
                et_distrib=self._normalize_distribution(dataset.review.recipe_uniform_distrib),
            ),
            described_as=edict(
                et="word",
                et_distrib=self._normalize_distribution(dataset.review.word_distrib),
            ),
            mentions=edict(
                et="word",
                et_distrib=self._normalize_distribution(dataset.review.word_distrib),
            ),
            belongs_to=edict(
                et="tag",
                et_distrib=self._normalize_distribution(dataset.belongs_to.et_distrib),
            ),
            has_ingredient=edict(
                et="ingredient",
                et_distrib=self._normalize_distribution(dataset.has_ingredient.et_distrib),
            ),
        )
        for relation in self.relations:
            embed = self._create_relation_embedding()
            setattr(self, relation, embed)
            bias = self._create_relation_bias(len(self.relations[relation].et_distrib))
            setattr(self, f"{relation}_bias", bias)

    def compute_loss(self, batch_indices):
        """Compute knowledge graph negative sampling loss.
        batch_indices: batch_size * 5 array, where each row is
                (u_id, r_id, w_id, t_id, i_id).
        """
        user_indices = batch_indices[:, 0]
        recipe_indices = batch_indices[:, 1]
        word_indices = batch_indices[:, 2]
        tag_indices = batch_indices[:, 3]
        ingredient_indices = batch_indices[:, 4]

        regularizations = []

        # user + view -> recipe
        ur_loss, ur_embeds = self._negative_sampling_loss(
            "user", "view", "recipe", user_indices, recipe_indices
        )
        regularizations.extend(ur_embeds)
        loss = ur_loss

        # user + mentions -> word
        uw_loss, uw_embeds = self._negative_sampling_loss(
            "user", "mentions", "word", user_indices, word_indices
        )
        regularizations.extend(uw_embeds)
        loss += uw_loss

        # recipe + described_as -> word
        rw_loss, rw_embeds = self._negative_sampling_loss(
            "recipe", "described_as", "word", recipe_indices, word_indices
        )
        regularizations.extend(rw_embeds)
        loss += rw_loss

        # recipe + belongs_to -> tag
        rt_loss, rt_embeds = self._negative_sampling_loss(
            "recipe", "belongs_to", "tag", recipe_indices, tag_indices
        )
        if rt_loss is not None:
            regularizations.extend(rt_embeds)
            loss += rt_loss

        # recipe + has_ingredient -> ingredient
        ri_loss, ri_embeds = self._negative_sampling_loss(
            "recipe", "has_ingredient", "ingredient", recipe_indices, ingredient_indices
        )
        if ri_loss is not None:
            regularizations.extend(ri_embeds)
            loss += ri_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = sum(torch.norm(term) for term in regularizations)
            loss += self.l2_lambda * l2_loss

        return loss