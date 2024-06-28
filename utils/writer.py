import configparser
import datetime
from torch.utils.tensorboard import SummaryWriter
import csv

class Writer:   
    date = datetime.date.today()
    time = datetime.datetime.now().strftime("%H-%M-%S")
    
    def __init__(self, is_test=False):
        if not is_test:
            self.writer = SummaryWriter(log_dir=f"log/{self.date}_{self.time}")
        
        config = configparser.ConfigParser()
        config.read("./config.cfg")
        self.customer_num = int(config["instance"]["customer_num"])
        
        self.episode = 0
        self.step = 0
        self.epoch = 0

        self.episode_reward_list = []
        self.step_reward_list = []
        self.test_cost_list = []
        self.test_time_list = []

    def episode_record(self, travel_time, penalty, reward):
        self.episode += 1
        self.writer.add_scalar('episode travel time', travel_time, self.episode)
        self.writer.add_scalar('episode penalty', penalty, self.episode)
        self.writer.add_scalar('episode reward', reward, self.episode)
        self.episode_reward_list.append(reward)
    
    def step_record(self):
        self.step += 1
        r = sum(self.episode_reward_list) / len(self.episode_reward_list)
        self.writer.add_scalar('step reward', r, self.step)
        self.step_reward_list.append(r)
        self.episode_reward_list = []

    def epoch_record(self):
        self.epoch += 1
        r = sum(self.step_reward_list) / len(self.step_reward_list)
        self.writer.add_scalar('epoch reward', r, self.epoch)
        self.step_reward_list = []

    def test_record(self, cost, time):
        self.test_cost_list.append(cost)
        self.test_time_list.append(time)
    
    def test_csv(self):
        data = list(zip(self.test_cost_list, self.test_time_list))
        with open('test_output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)  
            writer.writerow(['Cost', 'Time'])
            writer.writerows(data)
