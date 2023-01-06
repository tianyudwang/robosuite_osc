import datetime
import uuid
import io
import pathlib

import numpy as np

import robosuite as suite

from stable_baselines3 import PPO

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def load_episodes(directory, capacity=None):
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
    return episodes

def eplen(episode):
    return len(episode['action'])

def make_env(env_name):
    controller_configs = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=env_name, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        controller_configs=controller_configs,
    )
    return env


def sample_episodes(env_name, policy, expert_obs_keys, directory, num_episodes=1):

    env = make_env(env_name)

    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()
        obs_keys = list(obs.keys())
        done = False
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []

        while not done:         
            policy_obs = np.concatenate([obs[k] for k in expert_obs_keys])
            action, _ = policy.predict(policy_obs)
            obs, rew, done, info = env.step(action)
            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        if np.sum(episode['reward']) < 900:
            print("Discarding current episode")
            continue
        else:
            save_episode(directory, episode)
            episodes_saved += 1

def main():
    expert_model_filename = 'expert_models/PPO_Lift_OSC'
    expert_obs_keys = [
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
        # 'robot0_gripper_qvel',
        # 'cube_pos',
        # 'cube_quat',
        # 'gripper_to_cube_pos',
        'object-state',
    ]
    policy = PPO.load(expert_model_filename)
    env_name = "Lift"
    directory = pathlib.Path("./demonstrations/Lift/")
    sample_episodes(env_name, policy, expert_obs_keys, directory, num_episodes=64)

    # episodes = load_episodes(directory)




if __name__ == '__main__':
	main()