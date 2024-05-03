import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from f1tenth_drl.DataTools.MapData import MapData
from f1tenth_drl.Planners.TrackLine import TrackLine 

from f1tenth_drl.DataTools.plotting_utils import *

set_number = 1
map_name = "gbr"
base = f"Data/Experiment_{set_number}/"
names = ["Classic"]
sub_paths_td3 = [f"AgentOff_TD3_Game_gbr_Cth_8_1_1/"]

lap_n = 2
lap_list = np.ones(4, dtype=int) * lap_n


class TestLapData:
    def __init__(self, path, lap_n=0):
        self.path = path
        
        self.vehicle_name = self.path.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        self.map_name = self.vehicle_name.split("_")[3]
        self.map_data = MapData(self.map_name)
        self.std_track = TrackLine(self.map_name, False)
        self.racing_track = TrackLine(self.map_name, True)
        
        self.states = None
        self.actions = None
        self.lap_n = lap_n

        self.load_lap_data()

    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing{self.map_name.upper()}/Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def generate_state_progress_list(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.std_track.calculate_progress_percent(pt)
            # if p < progresses[-1]: continue
            progresses.append(p)
            
        return np.array(progresses[:-1]) * 100

    def calculate_deviations(self):
        # progresses = self.generate_state_progress_list()
        pts = self.states[:, 0:2]
        deviations = []
        for pt in pts:
            _, deviation = self.racing_track.get_cross_track_heading(pt)
            deviations.append(deviation)
            
        return np.array(deviations)

    def calculate_curvature(self, n_xs=200):
        thetas = self.states[:, 4]
        xs = np.linspace(0, 100, n_xs)
        theta_p = np.interp(xs, self.generate_state_progress_list(), thetas)
        curvatures = np.diff(theta_p) / np.diff(xs)
        
        return curvatures
    
    


def speed_distributions_algs():
    test_lap_data = TestLapData(base + sub_paths_td3[0], 10)  # Assuming lap number 0
    speeds = test_lap_data.states[:, 3]
    plt.figure(figsize=(10, 4))
    plt.hist(speeds, bins=np.arange(2, 8, 0.5), color='blue', alpha=0.7, density=True)
    plt.title("Speed Distribution of TD3 on 'mco' map")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
    #std_img_saving(name, True)

speed_distributions_algs()