
import numpy as np  
import robosuite as suite
from stable_baselines3 import PPO 


OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object-state',
]

def get_ppo_obs(obs):
    obs = np.concatenate([obs[k] for k in OBS_KEYS])
    return obs


if __name__ == "__main__":

    policy = PPO.load('./expert_models/PPO_Lift_OSC')

    env_info = {'env_name': 'Lift', 'robots': 'Panda', 'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2}}
    env = suite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    while True:
        print("Playing back random episode... (press ESC to quit)")

        obs = env.reset()
        done = False

        for i in range(200):
            policy_obs = get_ppo_obs(obs)
            action, _ = policy.predict(policy_obs)

            obs, rew, done, info = env.step(action)
            env.render()


