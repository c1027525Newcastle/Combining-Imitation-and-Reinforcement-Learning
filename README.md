# Combining Imitation and Reinforcement Learning to Surpass Human Performance
This repository is used for the work in my (Luca Albinescu) dissertation project for Newcastle University.

Some files in this code were created using the code from the paper Evans et al., "Comparing deep reinforcement learning architectures for autonomous racing" that is available on GitHub [https://github.com/BDEvan5/f1tenth_drl.git] and from the official F1TENTH repository f1tenth_gym available at [https://github.com/f1tenth/f1tenth_gym.git].


# Usage
This software was developed and run on an Apple MacBook Pro 14”, 2021 model, with the following hardware specifications:
• Processor (CPU): Apple M1 Pro with 8 cores (6 performance cores + 2 effi- ciency)
• Graphics (GPU): Apple M1 Pro 14-core
• Memory (RAM): 16GB
• Storage: 512GB SSD

## Installation
After cloning the git repository on the machine the following commands need to be run in the terminal:

- Create and activate virtual environment:
python3.9 -m venv venv 
source venv/bin/activate 

- Install required packages:
pip install -r requirements.txt
pip install -e .

## Train and Test the RL, IL or combined BCTD3 algorithms
- The main file that states what algorithm and what its hyperparameters are, is `experiments/Experiment.yaml`.
- After the algorithm wanted and its hyperparameters are set, the `f1tenth/run_experiments.py` file can be run to train an test the agent.
- Before choosing the `BC` algorithm, `f1tenth/ImitationAlgorithms/GenerateDataSet.py` file needs to be run to generate the data for the imitation learning agent. Inside `experiments/GenerateDataSet.yaml` it can be chosen how many laps are going to be created by the function.
- Before choosing `BCTD3` as the algorithm, the `BC` agent should be trained first as `BCTD3` relies on a pre-trained IL agent.


## Visualize Results
- All information gathered throughout training and also all the saved models can be found in the `Data/Experiment_{set_n}` folder.
- Before any other graphs can be created we first run the `f1tenth/DataTools/Statistics/BuildAgentDfs.py`, where `exp_n` should match the number of the experiment, file that will create a CSV file containing testing laps data for each algorithm.
- To create graphs for visualising MeanVelocity over Laps, Time over Laps or Progress over Laps the `f1tenth/DataTools/Statistics/MeanVelocityAndTimeOverLaps.py` file can be run. There are two arrays where the user can choose what algorithm from what experiment they want to plot.
- To create heatmaps that show Progress, Lap Number and Mean Velocity the file `f1tenth/DataTools/Statistics/HeatMap.py` can be run. Here as before the user can choose what algorithm from which experiment they want to visualise.
- To create Trajectory Heatmaps for each agent's test lap the function `f1tenth/DataTools/Statistics/GenerateTrajectoryImgs.py` is run. The user needs to only choose the experiment number they want to generate the images. The graphs are saved in an `Imgs` file inside the experiment folder.


## Extra Information
- All maps should be placed inside the `maps` folder. New maps (i.e. from the f1tenth_gym) can be used for training our algorithms by just placing all necessary files inside `maps` and running the custom `maps/map_reformat.py` to correctly format them. Different maps can be chosen by changing the `map_name` variable inside `experiments/Experiment.yaml`.
