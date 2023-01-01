

from lapal.utils import utils

def main():
	expert_policy = './expert_models/SAC_Lift_OSC'

	venv = utils.build_venv("Lift", 8)
	paths = utils.collect_demo_trajectories(venv, expert_policy, 32)

if __name__ == '__main__':
	main()