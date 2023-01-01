import sys, os
import os.path as osp
import time
from ruamel.yaml import YAML

import gym 
import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC

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

    if params['use_human_demo']:
        demo_filename = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../demonstrations/', params['env_name'], 'demo.hdf5'))
        params['human_demo_filename'] = demo_filename
        assert osp.exists(demo_filename), f'Demonstrations not stored at {demo_filename}'
    else:
        model_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../expert_models/', params['expert_policy']))
        assert osp.exists(model_path + '.zip'), f"Trained expert model not saved as {model_path}"
        params['expert_policy'] = model_path

    data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../data'))
    logdir = params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = osp.join(data_path, logdir)
    if params['suffix'] is None:
        params['suffix'] = params['discriminator']['reward_type'] + '_' + params['generator']['type']
    logdir += '_' + params['suffix']
    params['logdir'] = logdir

    print(params)

    # dump params
    os.makedirs(logdir, exist_ok=True)
    import yaml
    with open(osp.join(logdir, 'params.yml'), 'w') as fp:
        yaml.safe_dump(params, fp, sort_keys=False)

    ##################################
    ### SETUP ENV, DISCRIMINATOR, GENERATOR
    ##################################
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['which_gpu'])

    env = utils.make_robosuite_env(params['env_name'])
    if params['env_name'] in ['Door', 'Lift']:
        env = utils.make_robosuite_env(params['env_name'])
    else:
        env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]

    ################################################
    # Action encoder/decoder
    ################################################
    ac_vae_params = params['ac_vae']
    if ac_vae_params['use_ac_vae']:
        ac_vae = CVAE(
            params['ob_dim'],
            params['ac_dim'],
            ac_vae_params['latent_dim'],
            lr=ac_vae_params['learning_rate'],
            kl_coef=ac_vae_params['kl_coef'],
        )
    else:
        ac_vae = None

    ################################################
    # Discriminator
    ################################################
    disc_params = params['discriminator']
    disc_input_size = params['ob_dim']
    if ac_vae_params['use_ac_vae']:
        disc_input_size += ac_vae_params['latent_dim']
    else:
        disc_input_size += params['ac_dim']

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
    logger = configure(params['logdir'], ["stdout", "csv", "log", "tensorboard"])

    ################################################
    # Generator
    ################################################
    gen_params = params['generator']
    policy_kwargs = {}
    if disc_params['use_disc']:
        policy_kwargs.update(dict(reward=disc.reward))
    if ac_vae_params['use_ac_vae']:
        policy_kwargs.update(
            dict(
                latent_ac_dim=ac_vae_params['latent_dim'],
                ac_encoder=ac_vae.ac_encoder,
                ac_decoder=ac_vae.ac_decoder
            )
        )

    policy = SAC(
        "MlpPolicy",
        venv,
        learning_rate=gen_params['learning_rate'],
        buffer_size=1000000,
        learning_starts=gen_params['learning_starts'],
        batch_size=gen_params['batch_size'],
        gradient_steps=gen_params['gradient_steps'],
        policy_kwargs=dict(net_arch=[256, 256]),
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
        ac_vae, 
        disc, 
        policy, 
        logger
    )
    irl_model.train()

if __name__ == '__main__':
    main()
