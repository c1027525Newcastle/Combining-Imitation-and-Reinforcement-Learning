from f1tenth_model.td3 import TestTD3, TrainTD3
from f1tenth_model.sac import TestSAC,TrainSAC
from f1tenth_model.ddpg import TestDDPG, TrainDDPG
from f1tenth_model.ppo import PPO
from f1tenth_model.dqn import DQN
from f1tenth_model.a2c import A2C


def create_train_agent(run_dict, state_dim):
    action_dim = 2
    
    if run_dict.algorithm == "TD3":
        agent = TrainTD3(state_dim, action_dim)
    elif run_dict.algorithm == "SAC":
        agent = TrainSAC(state_dim, action_dim)
    elif run_dict.algorithm == "DDPG":
        agent = TrainDDPG(state_dim, action_dim)
    # elif run_dict.algorithm == "PPO":
    #     agent = PPO(state_dim, action_dim)
    # elif run_dict.algorithm == "DQN":
    #     agent = DQN(state_dim, action_dim)
    # elif run_dict.algorithm == "A2C":
    #     agent = A2C(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    
def create_test_agent(filename, directory, run_dict):
    
    if run_dict.algorithm == "TD3":
        agent = TestTD3(filename, directory)
    elif run_dict.algorithm == "SAC":
        agent = TestSAC(filename, directory)
    elif run_dict.algorithm == "DDPG":
        agent = TestDDPG(filename, directory)
    # elif run_dict.algorithm == "PPO":
    #     agent = PPO(state_dim, action_dim)
    # elif run_dict.algorithm == "DQN":
    #     agent = DQN(state_dim, action_dim)
    # elif run_dict.algorithm == "A2C":
    #     agent = A2C(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    