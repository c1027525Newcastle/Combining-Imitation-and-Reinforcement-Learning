import numpy as np
from f1tenth_drl.Planners.TrackLine import TrackLine


def select_architecture(run, conf):
    if run.algorithm == "BC":
        architecture = PlanningArchitectureBC(run.map_name)
    elif run.state_vector == "Game":
        architecture = PlanningArchitecture(run, conf)
    else:
        raise ValueError("Unknown state vector type: " + run.state_vector)
            
    return architecture


# Architecture for TD3
class PlanningArchitecture:
    def __init__(self, run, conf):
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer
        self.n_wpts = 10 
        self.n_beams = 20
        self.waypoint_scale = 2.5
        self.state_space = self.n_wpts * 2 + 2 + self.n_beams
        self.action_space = 2

        self.track = TrackLine(run.map_name, False)
    
    def transform_obs(self, obs):
        idx, dists = self.track.get_trackline_segment([obs['poses_x'][0], obs['poses_y'][0]])
        
        upcomings_inds = np.arange(idx, idx+self.n_wpts)
        if idx + self.n_wpts >= self.track.N:
            n_start_pts = idx + self.n_wpts - self.track.N
            upcomings_inds[self.n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        relative_pts /= self.waypoint_scale
        relative_pts = relative_pts.flatten()
        
        speed = obs['linear_vels_x'][0] / self.max_speed
        steering_angle = obs['steering_deltas'][0] / self.max_steer
        
        scan = np.array(obs['scans'][0]) 
        scan = np.clip(scan/10, 0, 1)
        
        motion_variables = np.array([speed, steering_angle])
        state = np.concatenate((scan, relative_pts.flatten(), motion_variables))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed)

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action


# Architecture for BC
class PlanningArchitectureBC:
    def __init__(self, map_name):
        self.max_speed = 8
        self.max_steer = 0.4
        self.n_wpts = 10 
        self.n_beams = 20
        self.waypoint_scale = 2.5
        self.state_space = self.n_wpts * 2 + 2 + self.n_beams
        self.action_space = 2

        self.track = TrackLine(map_name, False)
    
    def transform_obs(self, obs):
        idx, dists = self.track.get_trackline_segment([obs['poses_x'][0], obs['poses_y'][0]])
        
        upcomings_inds = np.arange(idx, idx+self.n_wpts)
        if idx + self.n_wpts >= self.track.N:
            n_start_pts = idx + self.n_wpts - self.track.N
            upcomings_inds[self.n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        relative_pts /= self.waypoint_scale
        relative_pts = relative_pts.flatten()
        
        speed = obs['linear_vels_x'][0] / self.max_speed
        steering_angle = obs['steering_deltas'][0] / self.max_steer
        
        scan = np.array(obs['scans'][0]) 
        scan = np.clip(scan/10, 0, 1)
        
        motion_variables = np.array([speed, steering_angle])

        # Uncomment to check sizes
        #print("\nShowing sizes to match datasets: ", len(scan), len(relative_pts), len(motion_variables), "\n")
        state = np.concatenate((scan, relative_pts.flatten(), motion_variables))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed)

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action
    
        
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts
