import numpy as np

from f1tenth.Planners.TrackLine import TrackLine


def select_reward_function(run, conf, std_track):
    reward = run.run_name.split("_")[4]
    if reward == "Cth":
        return CrossTrackHeadReward(std_track, conf)
    else:
        raise ValueError(f"Reward function not recognised: {reward}")
    

class CrossTrackHeadReward:
    def __init__(self, track: TrackLine, conf):
        self.track = track
        self.r_veloctiy = 1
        self.r_distance = 1
        self.max_v = conf.max_v # used for scaling.

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_counts'][0]:
            return 1  # complete
        if observation['collisions'][0]:
            return -1 # crash

        position = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        theta = observation['poses_theta'][0]
        speed = observation['linear_vels_x'][0]
        heading, distance = self.track.get_cross_track_heading(position)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (speed / self.max_v)

        r_distance = distance * self.r_distance 

        reward = r_heading - r_distance
        reward = max(reward, 0)
        return reward


def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))
