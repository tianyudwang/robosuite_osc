from typing import Optional, Union, Any, Tuple, List, Callable, Dict

import h5py
import json
import numpy as np
import torch as th

import gym 
from gym.envs import registry
import robosuite as suite
from robosuite.wrappers import GymWrapper
# from robosuite.utils.mjcf_utils import postprocess_model_xml

from stable_baselines3 import SAC, PPO
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from lapal.utils import pytorch_utils as ptu
from lapal.utils import types

ROBOSUITE_ENVS = ["Door", "Lift", "PickPlaceCan", "PickPlaceBread"]

def make_robosuite_env(env_name=None, obs_keys=None, controller_type='OSC_POSE'):
    controller_configs = suite.load_controller_config(default_controller=controller_type)
    env = suite.make(
        env_name=env_name, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        controller_configs=controller_configs,
        use_touch_obs=True,
    )
    if obs_keys is None:
        obs_keys = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'object-state',
        ]
    env = GymWrapper(env, keys=obs_keys)
    return env

def build_venv(env_name, n_envs=1, obs_keys=None, controller_type='OSC_POSE'):
    """
    Make vectorized env and add env wrappers
    """
    if env_name in ROBOSUITE_ENVS:
        env_kwargs = dict(
            env_name=env_name, 
            obs_keys=obs_keys, 
            controller_type=controller_type,
        )
        venv = make_vec_env(
            make_robosuite_env, 
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv, 
            n_envs=n_envs,
        )  
        return venv
    else:
        raise ValueError('Environment {} not supported yet ...'.format(env_name))
    return venv

def sample_trajectories(
    venv: VecEnv,
    policy: Union[OffPolicyAlgorithm, Any],
    num_trajectories: int,
    deterministic: Optional[bool] = True,
    ac_decoder: Callable[[np.ndarray], np.ndarray] = None,
) -> List[types.TrajectoryWithReward]:
    """
    Currently only works for fixed horizon envs, all envs should return done=True 
    """

    num_envs = venv.num_envs
    paths = []
    for i in range(num_trajectories // num_envs):
        obs, acs, rewards, next_obs, infos = [], [], [], [], []

        done = [False]
        ob = venv.reset()
        while not any(done):

            ac, _ = policy.predict(ob, deterministic=deterministic)

            if ac_decoder is not None:
                ob_tensor = ptu.from_numpy(ob)
                ac_tensor = ptu.from_numpy(ac)
                ac = ptu.to_numpy(ac_decoder(ob_tensor, ac_tensor))

            next_ob, reward, done, info = venv.step(ac)

            obs.append(ob)
            acs.append(ac)
            rewards.append(reward)
            next_obs.append(next_ob)
            infos.append(info)

            ob = next_ob

        obs = np.stack(obs, axis=0)
        acs = np.stack(acs, axis=0)
        next_obs = np.stack(next_obs, axis=0)
        rewards = np.stack(rewards, axis=0)

        for j in range(num_envs):
            if isinstance(infos[0][0], dict):
                keys = infos[0][0].keys()
                new_infos = {}
                for key in keys:
                    new_infos[key] = np.array([var[j][key] for var in infos])
            else:
                new_infos = None

            paths.append(
                types.TrajectoryWithReward(
                    observations=obs[:,j], 
                    actions=acs[:,j], 
                    next_observations=next_obs[:,j],
                    rewards=rewards[:,j],
                    infos=new_infos,
                    log_probs=None
                )
            )
    return paths


def check_demo_performance(paths):
    assert type(paths[0]) == types.TrajectoryWithReward, "Demo path type is not types.TrajectoryWithReward"
    returns = [path.rewards.sum() for path in paths]
    lens = [len(path) for path in paths]
    print(f"Collected {len(returns)} expert demonstrations")
    print(f"Demonstration length {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"Demonstration return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

    # print(f"Using {top_num} best trajectories")
    # new_paths = []
    # for i, ret in enumerate(returns):
    #     if ret > 900:
    #         new_paths.append(paths[i])
    # paths = new_paths[:top_num]

    # returns = [path.rewards.sum() for path in paths]
    # lens = [len(path) for path in paths]
    # print(f"Demonstration length {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    # print(f"Demonstration return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

def collect_demo_trajectories(env: gym.Env, expert_policy: str, batch_size: int):
    policy_cls = expert_policy.split('/')[-1].split('_')[0]
    if policy_cls == 'SAC':
        expert_policy = SAC.load(expert_policy)
    elif policy_cls == 'PPO':
        expert_policy = PPO.load(expert_policy)
    print('\nRunning expert policy to collect demonstrations...')
    demo_paths = sample_trajectories(env, expert_policy, batch_size)
    check_demo_performance(demo_paths, batch_size)
    return demo_paths


def load_episodes(directory, obs_keys, capacity=None):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        episodes[str(filename)] = episode

    paths = []
    for ep in episodes.values():
        obs = np.concatenate([ep[k] for k in obs_keys], axis=-1)
        paths.append(types.TrajectoryWithReward(
                observations=obs[:-1,:],
                actions=ep['action'],
                rewards=ep['reward'],
                next_observations=obs[1:,:],
                infos=None,
                log_probs=None,
            )
        )
    check_demo_performance(paths)
    return paths

def collect_human_trajectories(env_name, demo_filename):

    f = h5py.File(demo_filename, "r")   

    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    # TODO: Check env_cfg is the same as in demo 
    demo_env = make_robosuite_env(env_name=env_name)
    demos = list(f["data"].keys())

    paths = []
    for ep in demos:
        obs, acs, rewards, next_obs = [], [], [], []
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        demo_env.reset()
        xml = postprocess_model_xml(model_xml)
        demo_env.reset_from_xml_string(xml)
        demo_env.sim.reset()

        # load the flattened mujoco states
        states = np.array(f["data/{}/states".format(ep)][()])

        # load the initial state
        demo_env.sim.set_state_from_flattened(states[0])
        demo_env.sim.forward()
        ob = demo_env._get_observations(force_update=True)
        ob = np.concatenate([ob['object-state'], ob['robot0_proprio-state']])

        # load the actions and play them back open-loop
        actions = np.array(f["data/{}/actions".format(ep)][()])
        num_actions = actions.shape[0]

        rew = []
        done = False
        for j in range(1000):
            if j < num_actions:
                ac = actions[j]
                next_ob, reward, done, info = demo_env.step(ac)

                # ensure that the actions deterministically lead to the same recorded states
                if j < num_actions - 1:
                    state_playback = demo_env.sim.get_state().flatten()
                    if not np.allclose(states[j + 1], state_playback):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.6f} for ep {ep} at step {j}")  
            else:
                # use the last time_step to pad the episode
                # if j == self.cfg.env.horizon-1:
                #     time_step = time_step._replace(step_type=StepType.LAST)
                
                # TODO: null action to pad the episode depends on the task
                # change in position and orientation is 0, and grasp is 1
                pose_delta = np.zeros(6, dtype=np.float32)
                if env_name in ['Lift', 'Door']:
                    grasp = np.ones(1, dtype=np.float32)
                else:
                    grasp = np.zeros(1, dtype=np.float32)
                ac = np.concatenate((pose_delta, grasp))
                next_ob, reward, done, info = demo_env.step(ac)
            
            obs.append(ob)
            acs.append(ac)
            rewards.append(reward)
            next_obs.append(next_ob)

            ob = next_ob

        obs = np.stack(obs, axis=0)
        acs = np.stack(acs, axis=0)
        next_obs = np.stack(next_obs, axis=0)
        rewards = np.stack(rewards, axis=0)
        print(f"Current episode return: {np.sum(rewards):.2f}")

        paths.append(  
            types.TrajectoryWithReward(
                observations=obs, 
                actions=acs, 
                next_observations=next_obs,
                rewards=rewards,
                infos=None,
                log_probs=None
            )
        )

    return paths

def extract_paths(paths: List[types.Trajectory]) -> List[th.Tensor]:
    obs = ptu.from_numpy(np.array([path.observations for path in paths]))
    # Drop the last terminal state
    obs = obs[:, :-1, :]
    act = ptu.from_numpy(np.array([path.actions for path in paths]))
    if paths[0].log_probs is not None:
        log_probs = ptu.from_numpy(np.array([path.log_probs for path in paths]))
    else: 
        log_probs = None
    assert obs.shape[0] == act.shape[0], (
        "Batch size is not same for extracted paths"
    )
    assert obs.shape[1] == act.shape[1], (
        "Episode length is not same for extracted paths"
    )
    return obs, act, log_probs

def extract_transitions(transitions: List[types.Transition]) -> List[th.Tensor]:
    obs = ptu.from_numpy(np.array([transition.observation for transition in transitions]))
    act = ptu.from_numpy(np.array([transition.action for transition in transitions]))
    rew = ptu.from_numpy(np.array([transition.reward for transition in transitions]))
    if transitions[0].log_prob is not None:
        log_probs = ptu.from_numpy(np.array([transition.log_prob for transition in transitions]))
    else: 
        log_probs = None
    assert obs.shape[0] == act.shape[0], (
        "Batch size is not same for extracted paths"
    )
    return obs, act, rew, log_probs

def log_metrics(logger, metrics: Dict[str, np.ndarray], namespace: str):
    for k, v in metrics.items():
        if v.ndim < 1 or (v.dim == 1 and v.shape[0] <= 1):
            logger.record_mean(f"{namespace}/{k}", v)
        else:
            logger.record_mean(f"{namespace}/{k}Max", th.amax(v).item())
            logger.record_mean(f"{namespace}/{k}Min", th.amin(v).item())
            logger.record_mean(f"{namespace}/{k}Mean", th.mean(v).item())
            logger.record_mean(f"{namespace}/{k}Std", th.std(v).item())

def get_gym_env_type(env_name):
    if env_name not in registry.env_specs:
        raise ValueError("No such env")
    entry_point = registry.env_specs[env_name].entry_point
    if entry_point.startswith("gym.envs."):
        type_name = entry_point[len("gym.envs."):].split(":")[0].split('.')[0]
    else:
        type_name = entry_point.split('.')[0]
    return type_name
