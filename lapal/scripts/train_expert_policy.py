import argparse
import os
import os.path as osp
import numpy as np

import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from lapal.utils import utils

def build_env(env_name, n_envs, controller_type):
    """
    Make env and add env wrappers
    """

    if env_name in ["Door", "Lift"]:
        import robosuite as suite
        from robosuite.wrappers import GymWrapper

        def make_env():
            if controller_type == 'OSC_POSE':
                obs_keys = [
                    'robot0_eef_pos',
                    'robot0_eef_quat',
                    'robot0_gripper_qpos',
                ]
            elif controller_type == 'JOINT_VELOCITY':
                obs_keys = ['robot0_proprio-state']
            else:
                raise ValueError(f'{controller_type} controller type not supported.')
            controller_configs = suite.load_controller_config(default_controller=controller_type)
            obs_keys.append('object-state')
            env = suite.make(
                env_name=env_name, # try with other tasks like "Stack" and "Door"
                robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
                reward_shaping=True,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                controller_configs=controller_configs,
            )
            env = GymWrapper(env, keys=obs_keys)
            return env
        env = make_vec_env(make_env, vec_env_cls=SubprocVecEnv, n_envs=n_envs)  
        return env


    if env_name in ['Hoppper-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3']:
        env_kwargs = dict(terminate_when_unhealthy=False)
    else:
        env_kwargs = None

    if utils.get_gym_env_type(env_name) == 'mujoco':
        env = make_vec_env(
            env_name, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs,
            wrapper_class=wrapper, 
            wrapper_kwargs=wrapper_kwargs
        )

    else:
        raise ValueError(f'Environment {env_name} not supported yet ...')
    return env

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
    

    train_env = build_env(args.env_name, args.n_envs, args.controller_type)
    eval_env = build_env(args.env_name, args.n_envs, args.controller_type)
    print(f'Observation space: {train_env.observation_space}')
    print(f'Action space: {train_env.action_space}')

    policy_name = f"{args.algo}_{args.env_name}_{args.controller_type}"
    model = train_policy(train_env, eval_env, args.algo, policy_name, timesteps=args.total_timesteps)


if __name__ == '__main__':

    main()
