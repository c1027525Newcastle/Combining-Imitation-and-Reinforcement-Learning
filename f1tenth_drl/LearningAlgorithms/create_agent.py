from f1tenth_drl.LearningAlgorithms.td3 import TestTD3, TrainTD3


def create_train_agent(run_dict, state_dim):
    action_dim = 2
    
    if run_dict.algorithm == "TD3":
        agent = TrainTD3(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    
def create_test_agent(filename, directory, run_dict):
    
    if run_dict.algorithm == "TD3":
        agent = TestTD3(filename, directory)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    