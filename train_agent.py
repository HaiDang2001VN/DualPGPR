from __future__ import absolute_import, division, print_function

import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from kg_env import BatchKGEnvironment, BatchCFKGEnvironment
from utils import *
torch.autograd.set_detect_anomaly(True)

logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        self.saved_actions = []
        self.saved_cf_actions = []

        self.entropy = []
        self.cf_entropy = []
        
        self.rewards = []

    def forward(self, inputs):
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        x = self.l1(state)
        x = F.dropout(F.selu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.selu(out), p=0.5)
        actor_logits = self.actor(x)
        state_values = self.critic(x)  # Tensor of [bs, 1]
        act_mask = act_mask.bool()    
        actor_logits[~act_mask] = -999999.0
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]
            
        return act_probs, state_values

    def _select_action(self, batch_state, batch_act_mask, device, cf=False):
        """
        Selects an action based on the given batch state and action mask.

        Args:
            batch_state (list): A list of batch states.
            batch_act_mask (list): A list of action masks.
            device: The device to perform computations on.

        Returns:
            list: A list of selected actions.

        """
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.ByteTensor(batch_act_mask).to(device)  # Tensor of [bs, act_dim]

        probs, value = self((state, act_mask))  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        if cf:
            self.saved_cf_actions.append(SavedAction(m.log_prob(acts), value))
            self.cf_entropy.append(m.entropy())
        else:
            self.saved_actions.append(SavedAction(m.log_prob(acts), value))
            self.entropy.append(m.entropy())
            
        return acts.cpu().numpy().tolist()
    
    def select_action(self, batch_state, batch_act_mask, cf_batch_state, cf_batch_act_mask, device):
        batch_act_idx = self._select_action(batch_state, batch_act_mask, device)
        cf_batch_act_idx = self._select_action(cf_batch_state, cf_batch_act_mask, device, True)
        return batch_act_idx, cf_batch_act_idx

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.saved_cf_actions[:]
            del self.entropy[:]
            del self.cf_entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss, critic_loss, entropy_loss = 0, 0, 0
        cf_actor_loss, cf_critic_loss, cf_entropy_loss = 0, 0, 0

        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
            
            cf_log_prob, cf_value = self.saved_cf_actions[i]
            cf_advantage = batch_rewards[:, i] - cf_value.squeeze(1)
            cf_actor_loss += -cf_log_prob * cf_advantage.detach()
            cf_critic_loss += cf_advantage.pow(2)
            cf_entropy_loss += -self.cf_entropy[i]
            
        actor_loss, cf_actor_loss = actor_loss.mean(), cf_actor_loss.mean()
        critic_loss, cf_critic_loss = critic_loss.mean(), cf_critic_loss.mean()
        entropy_loss, cf_entropy_loss = entropy_loss.mean(), cf_entropy_loss.mean()

        loss = actor_loss + critic_loss + ent_weight * entropy_loss + cf_actor_loss + cf_critic_loss + ent_weight * cf_entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.saved_cf_actions[:]
        del self.entropy[:]
        del self.cf_entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


class ACDataLoader:
    def __init__(self, user_ids, batch_size):
        self.user_ids = np.array(user_ids)
        self.num_users = len(user_ids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_user_ids = self.user_ids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_user_ids.tolist()


def train(args):
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    cf_env = BatchCFKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    
    user_ids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(user_ids, args.batch_size)
    
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([name for name, _ in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        dataloader.reset()
        while dataloader.has_next():
            batch_user_ids = dataloader.get_batch()
            ### Start batch episodes ###
            batch_state = env.reset(batch_user_ids)  # numpy array of [bs, state_dim]
            cf_batch_state = cf_env.reset(batch_user_ids)  # numpy array of [bs, state_dim]
            done = False
            while not done:
                # Select action
                batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                cf_batch_act_mask = cf_env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                batch_act_idx, cf_batch_act_idx = model.select_action(batch_state, batch_act_mask, cf_batch_state, cf_batch_act_mask, args.device)  # int
                # Take step
                cf_batch_state, cf_batch_reward, _ = cf_env.batch_step(cf_batch_act_idx)
                batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                # Update state
                model.rewards.append(np.array([max(0.0, reward + cf_reward - 
                    F.cosine_similarity(torch.from_numpy(state), torch.from_numpy(cf_state), dim=0)) 
                    for reward, cf_reward, state, cf_state in zip(batch_reward, cf_batch_reward, batch_state, cf_batch_state)]))
            ### End of episodes ###

            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(user_ids) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight) # update model
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # Report performance
            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
                logger.info(
                        f'epoch/step={epoch}/{step}' +
                        f' | loss={avg_loss:.5f}' +
                        f' | ploss={avg_ploss:.5f}' +
                        f' | vloss={avg_vloss:.5f}' +
                        f' | entropy={avg_entropy:.5f}' +
                        f' | reward={avg_reward:.5f}')
        ### END of epoch ###

        policy_file = f'{args.log_dir}/policy_model_epoch_{epoch}.ckpt'
        logger.info(f"Save model to {policy_file}")
        torch.save(model.state_dict(), policy_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.5, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    args.log_dir = f'{TMP_DIR[args.dataset]}/{args.name}'
    os.makedirs(args.log_dir, exist_ok=True)

    global logger
    logger = get_logger(f'{args.log_dir}/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()