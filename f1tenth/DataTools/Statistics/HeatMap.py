import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from f1tenth.DataTools.Statistics.MeanVelocityAndTimeOverLaps import open_file


def heatmap(set_n, algorithm):
    lap_numbers, map_name, distances, progresses, lap_times, mean_velocities = open_file(set_n, algorithm)

    # Convert strings to numbers
    lap_numbers = [int(lap) for lap in lap_numbers]
    progresses = [float(progress) for progress in progresses] 
    mean_velocities = [float(velocities) for velocities in mean_velocities] 

    # Create a DataFrame from your lists
    data = pd.DataFrame({
        'Lap Number': lap_numbers,
        'Progress': progresses,
        'Mean Velocity': mean_velocities
    })

    # Sort the data by Lap Number
    data = data.sort_values(by='Lap Number', ascending=True)

    # Create a pivot table for the heatmap
    pivot_table = data.pivot(index="Lap Number", columns="Progress", values="Mean Velocity")

    # Convert pivot_table to numeric type
    pivot_table = pivot_table.astype(float)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".1f", linewidths=.5)
    plt.xlabel('Progress of lap(%)')
    plt.ylabel('Lap Number')
    # Add name to colorbar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Mean Velocity (m/s)')
    plt.show()


if __name__ == "__main__":
    # Experiment number and algorithm
    exp_n = 1
    algorithm = 'TD3'
    heatmap(exp_n, algorithm)