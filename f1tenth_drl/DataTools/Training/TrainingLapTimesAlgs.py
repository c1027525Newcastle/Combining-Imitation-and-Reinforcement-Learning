import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from f1tenth_drl.Utils.utils import *
from f1tenth_drl.DataTools.plotting_utils import *

def make_training_laptimes_plot():
    # Define the specific path to the file
    path = "Data/Experiment_1/AgentOff_TD3_Game_gbr_Cth_8_1_1/"
    
    # Load data from CSV
    rewards, lengths, progresses, _ = load_csv_data(path)
    
    # Calculate cumulative steps in kilometers
    steps = np.cumsum(lengths) / 1000
    
    # Initialize lists to collect lap times and corresponding step counts
    lap_times, completed_steps = [], []
    for i, progress in enumerate(progresses):
        if progress > 99:  # Check if the lap was completed
            lap_times.append(lengths[i]/10)  # Convert deciseconds to seconds
            completed_steps.append(steps[i])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(completed_steps, lap_times, '.', color=color_pallet[0], markersize=5, label='TD3', alpha=0.6, linewidth=2)
    
    # Customize plot appearance
    ax.set_title("Lap times over Training Steps", size=15)
    ax.set_xlabel("Training Steps (x1000)")
    ax.set_ylabel("Lap times (s)")
    ax.grid(True)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.legend()

    plt.show()

make_training_laptimes_plot()