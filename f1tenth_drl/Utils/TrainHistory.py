import numpy as np
from matplotlib import pyplot as plt
from f1tenth_drl.Utils.utils import *

SIZE = 20000


def plot_data(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    if len(values) >= moving_avg_period:
        moving_avg = true_moving_average(values, moving_avg_period)
        plt.plot(moving_avg)    
    if len(values) >= moving_avg_period*5:
        moving_avg = true_moving_average(values, moving_avg_period * 5)
        plt.plot(moving_avg)


class TrainHistory():
    def __init__(self, path) -> None:
        self.path = path

        self.ptr = 0
        self.lengths = np.zeros(SIZE)
        self.rewards = np.zeros(SIZE) 
        self.progresses = np.zeros(SIZE) 
        self.laptimes = np.zeros(SIZE) 
        self.t_counter = 0 
        
        self.ep_counter = 0
        self.ep_reward = 0
        self.reward_list = []
        

    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.reward_list.append(new_r)
        self.ep_counter += 1
        self.t_counter += 1 


    def lap_done(self, reward, progress, show_reward=False):
        self.add_step_data(reward)
        self.lengths[self.ptr] = self.ep_counter
        self.rewards[self.ptr] = self.ep_reward
        self.progresses[self.ptr] = progress
        self.ptr += 1
        self.reward_list = []

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []


    def print_update(self, plot_reward=False):
        if self.ptr < 10:
            return
        
        mean10 = np.mean(self.rewards[self.ptr-10:self.ptr])
        mean100 = np.mean(self.rewards[max(0, self.ptr-100):self.ptr])
        

    def save_csv_data(self):
        data = []
        ptr = self.ptr

        data.append(["Episode", "Reward", "Length", "Progress(%)", "Laptime"])

        for i in range(ptr): 
            if self.progresses[i] > 99 or (self.progresses[i] < 0.3 and self.laptimes[i] > 2):
                data.append([i, self.rewards[i], self.lengths[i], self.progresses[i], self.laptimes[i]])
            else:
                data.append([i, self.rewards[i], self.lengths[i], self.progresses[i], None])
        save_csv_array(data, self.path + "training_data_episodes.csv")

        t_steps = np.cumsum(self.lengths[0:ptr])/100
        
        plt.figure(3)
        
        plt.clf()
        plt.plot(t_steps, self.progresses[0:ptr], '.', color='darkblue', markersize=4)
        plt.plot(t_steps, true_moving_average(self.progresses[0:ptr], 20), linewidth='4', color='r')
        plt.ylabel("Average Progress")
        plt.xlabel("Training Steps (x100)")
        plt.ylim([0, 100])
        plt.grid(True)
        plt.savefig(self.path + "/training_progresses_steps.png")

        plt.clf()
        plt.plot(t_steps, self.rewards[0:ptr], '.', color='darkblue', markersize=4)
        plt.plot(t_steps, true_moving_average(self.rewards[0:ptr], 20), linewidth='4', color='r')
        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Reward per Episode")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.path + "/training_rewards_steps.png")
        
        plt.close(3)
