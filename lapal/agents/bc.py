

from tqdm import tqdm

import numpy as np
import torch as th
from torch import optim
from torch import nn

from lapal.utils.replay_buffer import ReplayBuffer
import lapal.utils.pytorch_utils as ptu
from lapal.utils import utils

from stable_baselines3.common.policies import ActorCriticPolicy

def lr_schedule(t):
    return 3e-4

class BC:
    def __init__(self, env, demo_paths, logger):

        self.env = env

        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.add_rollouts(demo_paths)

        # self.batch_size = 1
        self.logger = logger

        # policy_cfg = None
        self.policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=3e-4,
        )
        self.ent_weight = 1e-3

    def train(self):

        for i in tqdm(range(100000)):
            if i % 1000 == 0:
                self.evaluate()
            
            self.update()
            self.logger.dump(step=i)

    def evaluate(self):
        trajectories = utils.sample_trajectories(self.env, self.policy, 16)
        returns = np.array([np.sum(traj.rewards) for traj in trajectories])

        self.logger.record('Eval/return_mean', np.mean(returns))
        self.logger.record('Eval/return_max', np.amax(returns))
        self.logger.record('Eval/return_min', np.amin(returns))
        self.logger.record('Eval/return_std', np.std(returns))

        print(f"Eval return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

    def update(self):
        # trajectories_batch = self.replay_buffer.sample_random_trajectories(4)
        transitions = self.replay_buffer.sample_random_transitions(512)
        obs = np.stack([tran.observation for tran in transitions], axis=0)
        acts = np.stack([tran.action for tran in transitions], axis=0)
        obs, acts = ptu.from_numpy(obs), ptu.from_numpy(acts)
        
        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        # l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss #+ l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            'loss': loss.item(),
            'log_prob': log_prob.item(),
            'entropy': entropy.item(),
        }

        for k, v in metrics.items():
            self.logger.record(f'BC/{k}', v)

