import sys, os
import os.path as osp
import time
from ruamel.yaml import YAML

import robosuite as suite
from robosuite.wrappers import GymWrapper

from lapal.utils import utils
import lapal.utils.pytorch_utils as ptu
from lapal.agents.bc import BC

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_robosuite_env():
    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        # env_name="Door", # try with other tasks like "Stack" and "Door"
        env_name="Lift",
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # use_object_obs=True,
        controller_configs=controller_configs,
    )
    env = GymWrapper(env)
    return env

def build_venv(env_name, n_envs, norm_obs=False, wrapper=None, wrapper_kwargs=None):
    """
    Make vectorized env and add env wrappers
    """
    if env_name in ["Door", "Lift"]:
        env = make_vec_env(
            make_robosuite_env, 
            vec_env_cls=SubprocVecEnv, 
            n_envs=n_envs,
            wrapper_class=wrapper, 
            wrapper_kwargs=wrapper_kwargs
        )  
        return env

def main():

    yaml = YAML(typ='safe')
    params = yaml.load(open(sys.argv[1]))
    
    model_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../expert_models/', params['expert_policy']))
    assert osp.exists(model_path + '.zip'), f"Trained expert model not saved as {model_path}"
    params['expert_policy'] = model_path

    data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../../data'))
    logdir = params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = osp.join(data_path, logdir) + '_' + params['suffix']
    params['logdir'] = logdir

    # dump params
    # os.makedirs(logdir, exist_ok=True)
    # import yaml
    # with open(osp.join(logdir, 'params.yml'), 'w') as fp:
    #     yaml.safe_dump(params, fp, sort_keys=False)

    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['which_gpu'])

    venv = build_venv(params['env_name'], params['n_envs'])
    venv.seed(params['seed'])
    logger = configure(params['logdir'], ["tensorboard"])

    demo_paths = utils.collect_demo_trajectories(
        venv,
        params['expert_policy'], 
        params['demo_size']
    )


    bc_agent = BC(venv, demo_paths, logger)
    bc_agent.train()

if __name__ == '__main__':
    main()