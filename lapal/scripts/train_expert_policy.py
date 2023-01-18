import argparse
import os
import os.path as osp
import numpy as np

import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from lapal.utils import utils

def train_policy(env, eval_env, algo, policy_name, timesteps=100000):
    """
    Train the expert policy in RL
    """
    if algo == 'SAC':
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, verbose=1)
    if algo == 'DDPG':
        from stable_baselines3 import DDPG
        model = DDPG("MlpPolicy", env, train_freq=(4, "step"), verbose=1)  
    if algo == 'PPO':
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        raise ValueError('RL algorithm {} not supported yet ...'.format(algo))

    from stable_baselines3.common.logger import configure
    data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../data'))
    tmp_path = data_path + f'/{policy_name}'
    # set up logger
    new_logger = configure(tmp_path, ["stdout"])#, "csv", "log", "json", "tensorboard"])
    model.set_logger(new_logger)

    # callbacks
    save_freq = 10000
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=f"./{policy_name}/")
    # Separate evaluation env
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./{policy_name}/best_model",
                                 log_path=f"./{policy_name}/results", eval_freq=save_freq)
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_timesteps=timesteps, callback=callback)

    return model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Lift')
    parser.add_argument('--algo', type=str, default='SAC')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--total_timesteps', type=int, default=10000000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--controller_type', type=str, default='OSC_POSE')
    args = parser.parse_args()
    
    if args.controller_type == 'OSC_POSE':
        obs_keys = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ]
    elif args.controller_type == 'JOINT_VELOCITY':
        obs_keys = ['robot0_proprio-state']
    else:
        raise ValueError(f'{controller_type} controller type not supported.')
    obs_keys.append('object-state')

    train_env = utils.build_venv(args.env_name, args.n_envs, obs_keys, args.controller_type)
    eval_env = utils.build_venv(args.env_name, args.n_envs, obs_keys, args.controller_type)
    print(f'Observation space: {train_env.observation_space}')
    print(f'Action space: {train_env.action_space}')

    policy_name = f"{args.algo}_{args.env_name}_{args.controller_type}"
    model = train_policy(train_env, eval_env, args.algo, policy_name, timesteps=args.total_timesteps)


if __name__ == '__main__':

    main()
