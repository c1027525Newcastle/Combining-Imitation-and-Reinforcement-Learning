import numpy as np
import os
from f1tenth.Planners.Architectures import PlanningArchitectureBC
from sklearn.preprocessing import MinMaxScaler


class DataStateHistory:
    def __init__(self, experiment):
        self.path = experiment.folder + "RawData/"
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.vehicle_name = f"{experiment.planner_type}_{experiment.map_name}_{experiment.id_name}_{experiment.set_n}"
        self.states = []
        self.actions = []
        self.scans = []
    
        self.lap_n = 0

        # Use a matching architecture to TD3 to transform the observations and actions
        self.architecture = PlanningArchitectureBC(experiment.map_name)

    
    def add_memory_entry(self, obs, action):
        # Transform observation into the state using PlanningArchitecture logic
        transformed_state = self.architecture.transform_obs(obs)  
        # Transform action to match the used range
        transformed_action = self.architecture.transform_action(action)  
        self.states.append(transformed_state)
        self.actions.append(transformed_action)
        self.scans.append(obs['scans'][0])
    
    def save_history(self):
        states = np.array(self.states)
        actions = np.array(self.actions)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        # Scale both actions and states
        actions = scaler.fit_transform(actions)
        states = scaler.fit_transform(states)

        np.save(self.path + f"{self.vehicle_name}_Lap_{self.lap_n}_history.npy", states)
        np.save(self.path + f"{self.vehicle_name}_Lap_{self.lap_n}_actions.npy", actions)
        
        self.states = []
        self.actions = []
        self.scans = []
        self.lap_n += 1
