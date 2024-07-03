import configparser
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import csv

class Writer:   
    date = datetime.date.today()
    time = datetime.datetime.now().strftime("%H-%M-%S")
    test_file_path = 'test_output.csv'

    config = configparser.ConfigParser()
    config.read("./config.cfg")
    customer_num = int(config["instance"]["customer_num"])
    
    def __init__(self, is_test=False):
        if not is_test:
            self.writer = SummaryWriter(log_dir=f"log/{self.date}_{self.time}")
        
        self.episode = 0
        self.step = 0
        self.epoch = 0
        self.episode_reward_list = []
        self.step_reward_list = []

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

    def test_record(self, total_cost, travel_cost, penalty_cost,
                    service_status, time):
        service_status = service_status[1:] # delete depot
        early_service_list = [s for s in service_status if s[0] == -1]
        late_service_list = [s for s in service_status if s[0] == 1]
        tight_early_num = len([s for s in early_service_list if s[1] == True])
        loose_early_num = len([s for s in early_service_list if s[1] == False])
        tight_late_num = len([s for s in late_service_list if s[1] == True])
        loose_late_num = len([s for s in late_service_list if s[1] == False])
        data = [total_cost, travel_cost, penalty_cost, 
                tight_early_num, loose_early_num, tight_late_num, loose_late_num,
                time]
        
        if not os.path.exists(self.test_file_path):
            with open(self.test_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Total cost', 'Travel cost', 'Penalty cost', 
                                 'Number of customer served early (Tight)', 
                                 'Number of customer served early (Loose)', 
                                 'Number of customer served lately (Tight)', 
                                 'Number of customer served lately (Loose)', 
                                 'Time'])
                writer.writerow(data)
        else:
            with open(self.test_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
