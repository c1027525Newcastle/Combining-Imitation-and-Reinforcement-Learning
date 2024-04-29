from f1tenth_drl.LearningAlgorithms.td3 import TestTD3, TrainTD3
from f1tenth_drl.LearningAlgorithms.bctd3 import TestBCTD3, TrainBCTD3
from f1tenth_drl.ImitationAlgorithms.bc import TestBC, TrainBC
import numpy as np


def create_train_agent(run_dict, state_dim):
    action_dim = 2
    
    if run_dict.algorithm == "TD3":
        agent = TrainTD3(state_dim, action_dim)
    elif run_dict.algorithm == "BCTD3":
        agent = TrainBCTD3(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    
def create_test_agent(filename, directory, run_dict):
    if run_dict.algorithm == "TD3":
        agent = TestTD3(filename, directory)
    elif run_dict.algorithm == "BC":
        agent = TestBC()
    elif run_dict.algorithm == "BCTD3":
        agent = TestBCTD3()
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    