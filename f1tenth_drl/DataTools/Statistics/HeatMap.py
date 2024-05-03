import pandas as pd
import seaborn as sns # TODO: L add this to requirements
import matplotlib.pyplot as plt

from f1tenth_drl.DataTools.Statistics.MeanVelocityAndTimeOverLaps import open_file


def heatmap(set_n, algorithm):
    lap_numbers, map_name, distances, progresses, lap_times, mean_velocities = open_file(set_n, algorithm)

    # Create a DataFrame from your lists
    data = pd.DataFrame({
        'Lap Number': lap_numbers,
        'Progress': progresses,
        'Mean Velocity': mean_velocities
    })

    # Create a pivot table for the heatmap
    pivot_table = data.pivot(index="Lap Number", columns="Progress", values="Mean Velocity")

    # Convert pivot_table to numeric type
    pivot_table = pivot_table.astype(float)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".1f", linewidths=.5)
    plt.title('Mean Velocity by Lap Number and Progress')
    plt.xlabel('Progress (%)')
    plt.ylabel('Lap Number')
    # Add name to colorbar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Mean Velocity (m/s)')
    plt.show()


if __name__ == "__main__":
    set_n = 1
    algorithm = 'BCTD3'
    heatmap(set_n, algorithm)