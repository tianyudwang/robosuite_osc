import sys, os
import pathlib
import time
from ruamel.yaml import YAML

import gym 
import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC, PPO

from lapal.agents.discriminator import Discriminator
from lapal.agents.lapal_agent import LAPAL_Agent
from lapal.agents.vae import CVAE
from lapal.utils import utils
import lapal.utils.pytorch_utils as ptu


def main():

    yaml = YAML(typ='safe')
    params = yaml.load(open(sys.argv[1]))

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    expert_demo_dir = pathlib.Path(params['expert_folder']).resolve()

    data_path = pathlib.Path(__file__).parent.parent.parent / 'data'
    logdir = params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if params['suffix'] is None:
        params['suffix'] = params['discriminator']['reward_type'] + '_' + params['generator']['type']
    logdir += params['suffix']
    logdir = data_path / logdir
    params['logdir'] = str(logdir)

    print(params)

    # dump params
    logdir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(logdir / 'params.yml', 'w') as fp:
        yaml.safe_dump(params, fp, sort_keys=False)

    ##################################
    ### SETUP ENV, DISCRIMINATOR, GENERATOR
    ##################################
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['which_gpu'])

    if params['env_name'] in ['Door', 'Lift']:
        env = utils.make_robosuite_env(params['env_name'], params['obs_keys'])
    else:
        env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    ################################################
    # Discriminator
    ################################################
    disc_params = params['discriminator']
    disc_input_size = params['ob_dim'] + params['ac_dim']

    disc = Discriminator(
        input_size=disc_input_size,
        learning_rate=disc_params['learning_rate'],
        batch_size=disc_params['batch_size'],
        reward_type=disc_params['reward_type'],
        spectral_norm=disc_params.get('spectral_norm', False),
    )

    ################################################
    # Environment
    ################################################
    # SubprocVecEnv must be wrapped in if __name__ == "__main__":
    venv = utils.build_venv(params['env_name'], params['n_envs'])
    venv.seed(params['seed'])
    eval_venv = utils.build_venv(params['env_name'], params['n_envs'])
    eval_venv.seed(params['seed'] + 100)
    logger = configure(params['logdir'], ["stdout", "csv", "log", "tensorboard"])

    ################################################
    # Generator
    ################################################
    gen_params = params['generator']
    policy_kwargs = {}
    if disc_params['use_disc']:
        policy_kwargs.update(dict(reward=disc.reward))


    if gen_params['type'] == 'SAC':
        policy = SAC(
            "MlpPolicy",
            venv,
            learning_rate=gen_params['learning_rate'],
            buffer_size=1000000,
            learning_starts=gen_params['learning_starts'],
            batch_size=gen_params['batch_size'],
            gradient_steps=gen_params['gradient_steps'],
            policy_kwargs=gen_params['policy_kwargs'],
            seed=params['seed'],
            **policy_kwargs
        )
    elif gen_params['type'] == 'PPO':
        policy = PPO(
            "MlpPolicy",
            venv, 
            learning_rate=gen_params['learning_rate'],
            n_steps=gen_params['n_steps'],
            seed=params['seed'],
            **policy_kwargs
        )

    policy.set_logger(logger)

    print(f"Environment state space dimension: {params['ob_dim']}, action space dimension: {params['ac_dim']}")
    print(f"SAC policy state space dimension: {policy.policy.observation_space.shape[0]}, action space dimension: {policy.policy.action_space.shape[0]}")
    print(f"Discriminator input size: {disc_input_size}")

    ###################
    ### RUN TRAINING
    ###################

    irl_model = LAPAL_Agent(
        params, 
        venv, 
        eval_venv,
        disc, 
        policy, 
        logger,
        expert_demo_dir,
    )
    irl_model.train()

if __name__ == '__main__':
    main()
