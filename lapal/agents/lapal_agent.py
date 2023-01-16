import os
import os.path as osp

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from lapal.utils.replay_buffer import ReplayBuffer
from lapal.utils import utils, types
import lapal.utils.pytorch_utils as ptu

# from rl_plotter.logger import Logger

class LAPAL_Agent:
    def __init__(self, 
        params, 
        venv, 
        eval_venv,
        disc, 
        policy, 
        logger, 
        expert_demo_dir
    ):
        self.params = params

        self.logger = logger  
        self.venv = venv
        self.eval_venv = eval_venv
        self.disc = disc
        self.policy = policy 

        self.demo_buffer = ReplayBuffer()

        # bool
        self.use_disc = params['discriminator']['use_disc']

        # plot
        # self.rl_logger = Logger(
        #     log_dir='./logs',
        #     exp_name=params['suffix'],
        #     env_name=params['env_name'],
        #     seed=params['seed']
        # )

        self.expert_demo_dir = expert_demo_dir

    def train(self):
        # Run expert policy to collect demonstration paths

        if self.use_disc:
            demo_paths = utils.load_episodes(self.expert_demo_dir, self.params['obs_keys'])
            self.demo_buffer.add_rollouts(demo_paths)
            if params.get('init_agent_buffer_from_demo', False):
                self.load_demo_samples_to_agent()

        # Warm up generator replay buffer
        self.policy.learn(total_timesteps=self.params['generator']['learning_starts'])

        self.timesteps = 0
        while self.timesteps < self.params['total_timesteps']:
            if self.params['generator']['type'] == 'SAC':
                timesteps = self.params['generator']['batch_size']
            elif self.params['generator']['type'] == 'PPO':
                timesteps = self.params['generator']['n_steps'] * self.params['n_envs']
            self.train_generator(timesteps)

            if self.use_disc:
                self.train_discriminator()

            self.timesteps += timesteps

            # Evaluation
            if self.timesteps % self.params['evaluation']['interval'] < timesteps:
                self.perform_logging(self.policy)
            if self.timesteps % self.params['evaluation']['save_interval'] < timesteps:
                folder = self.save_models()
            self.logger.dump(step=self.timesteps)

        # save final model
        self.save_models()
        
    def load_demo_samples_to_agent(self):
        n_envs = self.params['n_envs']
        num_trajs = len(self.demo_buffer.trajectories)

        for i in range(num_trajs // n_envs):
            trajs = self.demo_buffer.trajectories[i*n_envs:(i+1)*n_envs]
            for t in range(len(trajs[0])):
                obs = np.stack([traj.observations[t] for traj in trajs], axis=0)
                next_obs = np.stack([traj.next_observations[t] for traj in trajs], axis=0)
                action = np.stack([traj.actions[t] for traj in trajs], axis=0) 
                reward = np.array([traj.rewards[t] for traj in trajs], dtype=np.float32)
                done = np.ones((n_envs,), dtype=np.float32) if t == len(trajs[0]) - 1 else np.zeros((n_envs,), dtype=np.float32) 
                infos = [{} for _ in range(n_envs)]
                self.policy.replay_buffer.add(obs, next_obs, action, reward, done, infos)
        print(f"Agent replay buffer contains {n_envs * self.policy.replay_buffer.pos} samples")

    # def pretrain_ac_vae(self):
    #     batch_size = self.params['ac_vae']['batch_size']
        
    #     for i in range(self.params['ac_vae']['n_iters']):
    #         demo_transitions = self.demo_buffer.sample_random_transitions(batch_size)
    #         obs = ptu.from_numpy(np.stack([t.observation for t in demo_transitions], axis=0))
    #         acs = ptu.from_numpy(np.stack([t.action for t in demo_transitions], axis=0))
    #         metrics = self.ac_vae.train(obs, acs)

    #         if (i + 1) % 1000 == 0:
    #             for k, v in metrics.items():
    #                 self.logger.record(f"ac_vae/{k}", v)
    #         else:
    #             for k, v in metrics.items():
    #                 self.logger.record(f"ac_vae/{k}", v, exclude='stdout')
            
    #         self.logger.dump(step=i)


    def train_generator(self, timesteps):
        """
        Train the policy/actor using learned reward
        """
        self.policy.learn(
            total_timesteps=timesteps, 
            reset_num_timesteps=False
        )


    def train_discriminator(self):
        batch_size = self.params['discriminator']['batch_size']  
        train_args = ()

        # Demo buffer contains ob, ac in original space
        demo_transitions = self.demo_buffer.sample_random_transitions(batch_size)
        demo_obs, demo_acs, demo_rews, _ = utils.extract_transitions(demo_transitions)
        train_args += (demo_obs, demo_acs,)

        # Agent buffer contains ob, ac in original space
        if self.params['generator']['type'] == 'SAC':
            agent_transitions = self.policy.replay_buffer.sample(batch_size)
        elif self.params['generator']['type'] == 'PPO':
            agent_transitions = next(self.policy.rollout_buffer.get(batch_size))
        agent_obs = agent_transitions.observations.float()
        agent_acs = agent_transitions.actions.float()
        train_args += (agent_obs, agent_acs,)

        metrics = self.disc.train(*train_args)
        metrics.update({'expert_true_rewards': th.mean(demo_rews).item()})

        for k, v in metrics.items():
            self.logger.record(f"disc/{k}", v)

    def perform_logging(self, eval_policy):

        #######################
        # Evaluate the agent policy in true environment
        print("\nCollecting data for eval...")
        eval_paths = utils.sample_trajectories(
            self.eval_venv, 
            eval_policy, 
            self.params['evaluation']['batch_size'],
        )  

        eval_returns = [path.rewards.sum() for path in eval_paths]
        eval_ep_lens = [len(path) for path in eval_paths]

        logs = {}
        logs["Eval/AverageReturn"] = np.mean(eval_returns)
        logs["Eval/StdReturn"] = np.std(eval_returns)
        logs["Eval/MaxReturn"] = np.max(eval_returns)
        logs["Eval/MinReturn"] = np.min(eval_returns)
        logs["Eval/AverageEpLen"] = np.mean(eval_ep_lens)

        for key, value in logs.items():
            self.logger.record(key, value)

        # self.rl_logger.update(score=eval_returns, total_steps=self.timesteps)

    def save_models(self):   

        folder = self.params['logdir'] + f"/checkpoints/{self.timesteps:08d}"
        os.makedirs(folder, exist_ok=True) 

        self.policy.save(osp.join(folder, "policy.pt"), exclude=['reward'])
        th.save(self.disc.state_dict(), osp.join(folder, "disc.pt"))

        return folder

    def load_models(self, folder): 

        print(f'Loading models from checkpoint {folder}')

        self.disc.load_state_dict(th.load(folder + "/disc.pt"))
        self.policy = SAC.load(folder + "/policy.pt")
        self.policy.set_env(self.venv)
        
