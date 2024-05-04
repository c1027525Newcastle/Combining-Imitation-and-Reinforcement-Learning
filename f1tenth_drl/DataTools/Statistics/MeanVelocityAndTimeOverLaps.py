import matplotlib.pyplot as plt
import numpy as np
import os

def open_file(set_n, algorithm):
    file_name = f'Data/Experiment_{set_n}/AgentOff_{algorithm}_Game_gbr_Cth_8_{set_n}_1/MainAgentData.csv'

    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = file.readlines()[1:]
            data = [line.strip().split(',') for line in data]
    else:
        print(f"File {file_name} does not exist")

    lap_numbers =[]
    distances = []
    progresses = []
    lap_times = []
    mean_velocities = []
    for line in data:
        lap_numbers.append(line[0])
        map_name = line[1]
        distances.append(line[2])
        progresses.append(line[3])
        lap_times.append(line[4])
        mean_velocities.append(line[5])

    return lap_numbers, map_name, distances, progresses, lap_times, mean_velocities


def mean_velocity_laps(set_n, algorithm):
    lap_numbers, map_name, distances, progresses, lap_times, mean_velocities = open_file(set_n, algorithm)

    # Convert strings to numbers
    lap_numbers = [int(lap) for lap in lap_numbers] 
    mean_velocities = [float(velocities) for velocities in mean_velocities]

    plt.plot(lap_numbers, mean_velocities)
    plt.ylim(top=8)
    plt.xlabel('Lap number')
    plt.ylabel('Mean velocity (m/s)')
    # Define the limits
    plt.ylim(bottom=5.5, top=8)
    # Define the ticks
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(6.25, 8, 0.25))
    plt.title(f'Mean velocity over laps for {algorithm} in map {map_name}')
    plt.show()


def time_over_laps(set_n, algorithm):
    lap_numbers, map_name, distances, progresses, lap_times, mean_velocities = open_file(set_n, algorithm)

    # Convert strings to numbers
    lap_numbers = [int(lap) for lap in lap_numbers] 
    lap_times = [float(times) for times in lap_times]

    progress_points = ['green' if progress >= '0.95' else 'red' for progress in progresses]
    plt.scatter(lap_numbers, lap_times, c=progress_points)
    plt.xlabel('Lap number')
    plt.ylabel('Lap Time (s)')
    # Define the limits
    plt.ylim(bottom=0, top=80)
    # Define the ticks
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 81, 20))
    plt.title(f'Lap Times over laps for {algorithm} in map {map_name}')
    #plt.show()

    plt.savefig(f'f1tenth_drl/DataTools/Statistics/{set_n}_{algorithm}_time_over_laps.png')

    # Close the plot
    plt.close()


def progress(set_n, algorithm):
    lap_numbers, map_name, distances, progresses, lap_times, mean_velocities = open_file(set_n, algorithm)

    # Convert strings to numbers
    lap_numbers = [int(lap) for lap in lap_numbers] 

    progress_points = ['green' if progress >= '0.95' else 'red' for progress in progresses]
    progresses = [float(progress) for progress in progresses]
    plt.scatter(lap_numbers, progresses, c=progress_points)
    plt.xlabel('Lap number')
    plt.ylabel('Progress (%)')
    # Define the limits
    plt.ylim(bottom=0, top=1.1)
    # Define the ticks
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(f'Lap Times over laps for {algorithm} in map {map_name}')
    
    #plt.show()

    plt.savefig(f'f1tenth_drl/DataTools/Statistics/{set_n}_{algorithm}_progress.png')
    # Close the plot
    plt.close()


if __name__ == "__main__":
    set_n = [1, 2, 3]
    algorithms = ['BC', 'TD3', 'BCTD3']
    #mean_velocity_laps(set_n, algorithm)
    #time_over_laps(set_n, algorithm)
    #progress(set_n, algorithm)

    # for algo in algorithms:
    #     for n in set_n:
    #         progress(n, algo)