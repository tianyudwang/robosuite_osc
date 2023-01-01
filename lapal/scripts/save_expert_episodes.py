import datetime
import uuid
import numpy as np

import robosuite as suite

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
    
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    return env

def sample_episodes(env, policy, num_episodes=1, directory='./demonstrations/SAC/Lift'):

    for i in range(num_episodes):
        obs = env.reset()
        obs_keys = list(obs.keys())
        done = False
        episode = {}

        while not done:
            policy_obs = 
            action, _ = SAC.predict(policy_obs)


        save_episode(directory, episode)




def main():
    expert_model_filename = 'expert_models/SAC_Lift'
    policy = SAC.load(expert_model_filename)

    env = make_env()




if __name__ == '__main__':
	main()