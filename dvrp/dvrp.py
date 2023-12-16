import numpy as np
import random
import copy

import event as et

class DVRP:
    time_upper_bound = 100000

    def __init__(self, file_path, dynamic_degree, update_interval):
        self.dynamic_degree = dynamic_degree
        self.update_interval = update_interval
        
        # Original data
        data = self.read_file(file_path)
        self.customer_num = data[0]
        self.customer_data = data[1]
        self.vehicle_num = data[2]
        self.vehicle_capacity = data[3]
        
        # Online data
        self.travel_time_parameter = self.make_travel_time_parameter([0.1, 0.6])
        online_data = self.init_online_data()
        self.online_customer_data = online_data[0]
        self.online_vehicle_data = online_data[1]
        self.online_travel_time_data = online_data[2]

        # Events
        self.event_manager = et.EventManager()
        self.init_dynamic_event()

    def read_file(self,file_path):
        """
        Read txt file to generate instance info.
        """

        customer_num = 0
        customer_data = []
        
        with open(file_path, 'r') as file:       
            for i, line in enumerate(file):
                if i in [0, 1, 2, 3, 5, 6, 7, 8]:
                    continue
                
                words = line.split()
                if i == 4:
                    vehicle_num = int(words[0])
                    vehicle_capacity = int(words[1])
                elif i > 8:
                    id = int(words[0])
                    x = int(words[1])
                    y = int(words[2])
                    demand = int(words[3])
                    earliest_time = int(words[4])
                    latest_time = int(words[5])
                    service_time = int(words[6])

                    customer_num += 1
                    customer_data.append([id, x, y, demand, earliest_time, latest_time, 
                                          service_time])

        customer_data = np.array(customer_data)

        return [customer_num, customer_data, vehicle_num, vehicle_capacity]

    def make_travel_time_parameter(self, mul_range):
        dist_matrix = np.zeros((self.customer_num, self.customer_num), dtype=float)
        for i in range(self.customer_num):
            for j in range(self.customer_num):
                if i == j:
                    continue
                elif i > j:
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    distance = ((self.customer_data[i, 0] - self.customer_data[j, 0])**2 +\
                                (self.customer_data[i, 1] - self.customer_data[j, 1])**2)**0.5
                    dist_matrix[i, j] = distance

        travel_time_mean = copy.deepcopy(dist_matrix)
        multiplier = np.random.uniform(mul_range[0], mul_range[1], travel_time_mean.shape)
        travel_time_std = travel_time_mean * multiplier
        travel_time_parameter = np.stack((travel_time_mean, travel_time_std), axis=2)

        return travel_time_parameter

    def init_online_data(self):
        # Customer
        c_data = copy.deepcopy(self.customer_data)
        add_column = np.zeros((c_data.shape[0], 2), dtype=int)
        c_data = np.hstack((c_data, add_column))
        self.reset_request_time(c_data)

        # Vehicle
        v_data = []
        for _ in range(self.vehicle_num):
            vehicle_status = [0, 0, self.vehicle_capacity, [], []]
            v_data.append(vehicle_status)

        # Travel time
        t_data = self.sample_travel_time()

        return [c_data, v_data, t_data]

    def reset_request_time(self, data):
        dynamic_num = int((self.customer_num-1) * self.dynamic_degree)
        dynamic_request = random.sample(range(1, self.customer_num), dynamic_num)
        for idx in dynamic_request:
            request_time = random.sample(range(1, self.customer_data[idx, 4]), 1)
            data[idx, -2] = request_time[0]

    def sample_travel_time(self):
        travel_time_matrix = np.zeros((self.customer_num, self.customer_num), dtype=float)
        for i in range(self.customer_num):
            for j in range(self.customer_num):
                mean = self.travel_time_parameter[i, j, 0]
                std = self.travel_time_parameter[i, j, 1]
                travel_time = np.random.normal(mean, std, 1)
                travel_time_matrix[i, j] = travel_time
        
        return travel_time_matrix
    
    def init_dynamic_event(self):
        # Dynamic customer request
        for i in range(self.customer_num):
            if self.online_customer_data[i, 7] > 0:
                time = self.online_customer_data[i, 7]
                id = self.online_customer_data[i, 0]
                event = et.ArrivalEvent(time, id)
                self.event_manager.add_event(event)

        # Dynamic travel time update
        t = 0 + self.update_interval
        while t < self.time_upper_bound:
            event = et.TravelTimeUpdateEvent(t)
            self.event_manager.add_event(event)
            t += self.update_interval

if __name__ == '__main__':
    file_path = "./instance/Vrp-Set-Solomon/C101.txt"
    prob = DVRP(file_path, 0.1, 20)
    