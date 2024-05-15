import numpy as np

from f1tenth_drl.f1tenth_gym import F110Env
from f1tenth_drl.Planners.AgentTrainer import AgentTrainer
from f1tenth_drl.Planners.AgentTester import AgentTester
from f1tenth_drl.Utils.utils import *
import torch


def run_simulation_loop_steps(env, planner, steps, steps_per_action=10):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for i in range(steps):
        action = planner.plan(observation)
        
        mini_i = steps_per_action
        while mini_i > 0:
            observation, reward, done, info = env.step(action[None, :])
            mini_i -= 1
        
            if done:
                planner.done_callback(observation)
                observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
                break

        
def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    for lap in range(n_laps):
        observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
        while not done:
            action = planner.plan(observation)
            
            mini_i = n_sim_steps
            while mini_i > 0 and not done:
                observation, reward, done, info = env.step(action[None, :])
                mini_i -= 1
    
        planner.done_callback(observation)
    

def seed_randomness(run_dict):
    random_seed = run_dict.random_seed + 10 * run_dict.n
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    
    
def run_training_batch(experiment):
    # run_list = setup_run_list(experiment, new_run=False)
    run_list = setup_run_list(experiment)
    conf = load_conf("config_file")
    
    for i, run_dict in enumerate(run_list):
        seed_randomness(run_dict)
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F110Env(map=run_dict.map_name, num_agents=1)
        
        # BC is trained in its own file so this functions needs only to run the testing part
        if run_dict.algorithm != "BC":
            # Added a way to allow interrupting training early to ease testing of code
            try:
                planner = AgentTrainer(run_dict, conf)
                print("Training")
                run_simulation_loop_steps(env, planner, run_dict.training_steps, 4)
                
                print("Testing")
                planner = AgentTester(run_dict, conf)
                run_simulation_loop_laps(env, planner, run_dict.n_test_laps, 4)
                env.__del__()

            except KeyboardInterrupt:
                print("Training interrupted")
                print("Testing")
                planner = AgentTester(run_dict, conf)
                run_simulation_loop_laps(env, planner, run_dict.n_test_laps, 4)
                env.__del__()
        elif run_dict.algorithm == "BC":
            print("Testing BC")
            planner = AgentTester(run_dict, conf)
            run_simulation_loop_laps(env, planner, run_dict.n_test_laps, 4)
            env.__del__()
        else:
            raise ValueError("Algorithm not recognised")

    
def main():
    experiment = "Experiment"
    run_training_batch(experiment)

    
if __name__ == "__main__":
    main()
