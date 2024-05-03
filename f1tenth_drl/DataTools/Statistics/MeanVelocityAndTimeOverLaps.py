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

    plt.plot(lap_numbers, mean_velocities)
    plt.ylim(top=8)
    plt.xlabel('Lap number')
    plt.ylabel('Mean velocity (m/s)')
    plt.title(f'Mean velocity over laps for {algorithm} in map {map_name}')
    plt.show()


def time_over_laps(set_n, algorithm):
    lap_numbers, map_name, distances, progresses, lap_times, mean_velocities = open_file(set_n, algorithm)

    progress_points = ['green' if progress >= '0.95' else 'red' for progress in progresses]
    plt.scatter(lap_numbers, lap_times, c=progress_points)
    plt.xlabel('Lap number')
    plt.ylabel('Lap Times (s)')
    plt.title(f'Lap Times over laps for {algorithm} in map {map_name}')
    plt.show()


if __name__ == "__main__":
    set_n = 1
    algorithm = 'TD3'
    mean_velocity_laps(set_n, algorithm)
    #time_over_laps(set_n, algorithm)