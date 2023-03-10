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

    expert_demo_dir = pathlib.Path(params['expert_folder']) / params['env_name'] / params['controller_type']
    expert_demo_dir = expert_demo_dir.resolve()

    data_path = pathlib.Path(__file__).parent.parent.parent / 'data' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['discriminator']['reward_type'],
        params['generator']['type'],
        params['controller_type'],
    ])
    if params['suffix'] is not None:
        logdir += '_' + params['suffix']
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

    ################################################
    # Environment
    ################################################
    # SubprocVecEnv must be wrapped in if __name__ == "__main__":
    venv = utils.build_venv(
        params['env_name'], 
        n_envs=params['n_envs'], 
        obs_keys=params['obs_keys'], 
        controller_type=params['controller_type'],
    )
    params['ob_dim'] = venv.observation_space.shape[0]
    params['ac_dim'] = venv.action_space.shape[0]
    eval_venv = utils.build_venv(
        params['env_name'], 
        n_envs=params['evaluation']['n_envs'], 
        obs_keys=params['obs_keys'],
        controller_type=params['controller_type'],
    )
    venv.seed(params['seed'])
    eval_venv.seed(params['seed'] + 100)
    logger = configure(params['logdir'], ["stdout", "csv", "log", "tensorboard"])

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
        spectral_norm=disc_params['spectral_norm'],
    )
    
    ################################################
    # Generator
    ################################################
    gen_params = params['generator']
    gen_kwargs = {}
    if disc_params['use_disc']:
        gen_kwargs.update(dict(reward=disc.reward))


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
            **gen_kwargs,
        )
    elif gen_params['type'] == 'PPO':
        policy = PPO(
            "MlpPolicy",
            venv, 
            learning_rate=gen_params['learning_rate'],
            n_steps=gen_params['n_steps'],
            seed=params['seed'],
            **gen_kwargs,
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
