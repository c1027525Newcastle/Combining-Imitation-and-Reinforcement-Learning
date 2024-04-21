from f1tenth_drl.LearningAlgorithms.td3 import TestTD3, TrainTD3
from f1tenth_drl.ImitationAlgorithms.bc import TestBC, TrainBC
import numpy as np


def create_train_agent(run_dict, state_dim):
    action_dim = 2
    
    if run_dict.algorithm == "TD3":
        agent = TrainTD3(state_dim, action_dim)
    elif run_dict.algorithm == "BC":
        scans_file = 'Data/GenerateDataSet_1/RawData/PurePursuit_mco_DataGen_1_Lap_0_scans.npy'
        actions_file = 'Data/GenerateDataSet_1/RawData/PurePursuit_mco_DataGen_1_Lap_0_history.npy'
        agent = TrainBC(scans_file, actions_file)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    
def create_test_agent(filename, directory, run_dict):
    
    if run_dict.algorithm == "TD3":
        agent = TestTD3(filename, directory)
    elif run_dict.algorithm == "BC":
        agent = TestBC()
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    