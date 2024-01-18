import numpy as np
from scipy.stats import gamma
import random
import copy

import model.event as et

class Env:
    gamma_shape_parameter = 1
    gamma_scale_parameter = 1

    def __init__(self, file_path, dynamic_degree, update_interval):
        self.dynamic_degree = dynamic_degree
        self.update_interval = update_interval
        
        # raw data
        data = self.read_file(file_path)
        self.customer_num = data[0]
        self.customer_data = data[1]
        self.vehicle_num = data[2]
        self.vehicle_capacity = data[3]

        # processed data
        self.dist_matrix = self.cal_dist_matrix()
        self.online_customer_data = None
        self.online_travel_time_data = None

    def read_file(self, file_path):
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

    def cal_dist_matrix(self):
        """
        Calculate the distance between customer
        """

        dist_matrix = np.zeros((self.customer_num, self.customer_num), dtype=float)
        for i in range(self.customer_num):
            for j in range(self.customer_num):
                if i == j:
                    continue
                elif i > j:
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    distance = ((self.customer_data[i, 1] - self.customer_data[j, 1])**2 +\
                                (self.customer_data[i, 2] - self.customer_data[j, 2])**2)**0.5
                    dist_matrix[i, j] = distance
        
        return dist_matrix

    def reset_request_time(self):
        """
        Change a certain proportion of customers to dynamically arrive
        """

        # change customer data shape
        c_data = copy.deepcopy(self.customer_data)
        add_column = np.zeros((c_data.shape[0], 2), dtype=int) # request time, status
        c_data = np.hstack((c_data, add_column))
        c_data[0, 8] = -1 # depot

        # set request time
        dynamic_num = int((self.customer_num-1) * self.dynamic_degree)
        dynamic_request = random.sample(range(1, self.customer_num), dynamic_num)
        for idx in dynamic_request:
            request_time = random.sample(range(1, self.customer_data[idx, 4]), 1)
            c_data[idx, 7] = request_time[0]
        
        return c_data

    def sample_travel_time(self):
        """
        Random sampling of travel times with gamma distribution
        """

        t_data = np.zeros((self.customer_num, self.customer_num), dtype=float)
        for i in range(self.customer_num):
            for j in range(self.customer_num):
                if self.dist_matrix[i, j] == 0:
                    t_data[i, j] = 0
                else:
                    t_data[i, j] = gamma.rvs(a = self.dist_matrix[i, j] * self.gamma_shape_parameter, 
                                             scale = self.gamma_scale_parameter)
                        
        return t_data
    
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

    def reset(self):
        # customer 
        self.online_customer_data = self.reset_request_time()

        # travel time
        self.online_travel_time_data = self.sample_travel_time()

    def get_customer_data(self):
        mask_data = copy.deepcopy(self.online_customer_data)
        rows_to_mask = np.where(mask_data[:, 7] > 0)[0]
        mask_data[rows_to_mask, :] = -1

        return mask_data

    def get_travel_time_data(self):
        return self.dist_matrix
        return self.online_travel_time_data

    def get_vehicle_capacity(self):
        return self.vehicle_capacity
