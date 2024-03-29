import numpy as np
from scipy.stats import gamma
import random
import copy
import heapq

import model.event as et

class Env:
    gamma_shape_parameter = 1
    gamma_scale_parameter = 1
    clock = 0

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
        self.online_vehicle_data = None
        self.online_travel_time_data = None
        self.event_list = None

        # event manager
        self.event_manager = et.EventManager()

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

    def init_dynamic_customer(self):
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

    def init_vehicle_data(self):
        """
        6 features of the vehicle: assigned route, completed route, 
        remaining capacity, completed node-to-node distance, completed service time.
        """

        v_data = []
        for _ in range(self.vehicle_num):
            vehicle_data = [[], [], self.vehicle_capacity, 0, 0]
            v_data.append(vehicle_data)
        
        return v_data

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
    
    def set_dynamic_event(self):
        """
        Return a list to store dynamic events and their time points.
        """
        
        event_list = []
        # dynamic customer request
        for i in range(self.customer_num):
            if self.online_customer_data[i, 7] > 0:
                t = self.online_customer_data[i, 7]
                id = self.online_customer_data[i, 0]
                heapq.heappush(event_list, (t, id))

        # dynamic travel time update
        if self.update_interval > 0:
            depot_close_time = self.online_customer_data[0, 5]
            t = self.update_interval
            while t < depot_close_time:
                heapq.heappush(event_list, (t, -1))
                t += self.update_interval
        
        # the close of depot
        close_time = self.online_customer_data[0, 5]
        heapq.heappush(event_list, (close_time, 0))

        return event_list

    def get_customer_data(self):
        mask_data = copy.deepcopy(self.online_customer_data)
        rows_to_mask = np.where(mask_data[:, 7] > 0)[0]
        mask_data[rows_to_mask, :] = -1

        return mask_data
    
    def get_vehicle_capacity(self):
        capacity_list =  [row[2] for row in self.online_vehicle_data]
        
        return capacity_list

    def get_state(self):
        decision_data = None
        c_data = self.get_customer_data()
        t_data = self.online_travel_time_data
        capacity = self.get_vehicle_capacity()
        
        return (decision_data, [c_data, t_data, capacity])

    def set_vehicle_data(self, routes):
        """
        Update the vehicle data and customer data according to the rotues.
        """

        for i in range(len(self.online_vehicle_data)):
            if i < len(routes):
                self.online_vehicle_data[i][0] = routes[i]
                for c in routes[i]:
                    if c == 0:
                        continue
                    self.online_customer_data[c, 8] = 1
            else:
                self.online_vehicle_data.pop()

    def advance_time(self):
        event = heapq.heappop(self.dynamic_event_list)
        print(f"event {event[1]} at {event[0]}")
        for v in self.online_vehicle_data:
            print("======================")
            if not v[1]:    # start the route
                v[1].append(v[0].pop(0))
            
            current_time = self.clock
            next_time = event[0]
            while(current_time <= next_time):
                time_interval = next_time - current_time
                if v[3] == 0: # serve customer now
                    current_node = v[1][-1]
                    if v[4] + time_interval > self.online_customer_data[current_node, 6]: # finish service and start travel
                        # service
                        used_time = self.online_customer_data[current_node, 6] - v[4]
                        current_time += used_time
                        v[4] = 0
                        print(f"Finish customer {current_node} at {current_time}")
                        
                        # travel
                        if not v[0]:    # no target 
                            break
                        
                        v[1].append(v[0].pop(0))
                        self.online_customer_data[v[1][-1], 8] = -1
                        v[2] -= self.online_customer_data[v[1][-1], 3]
                        current_node = v[1][-2]
                        next_node = v[1][-1]
                        speed = self.dist_matrix[current_node, next_node] / self.online_travel_time_data[current_node, next_node]
                        left_time = next_time - current_time
                        if speed * left_time > self.dist_matrix[current_node, next_node]: # have enough time to finsh traveling
                            v[3] = self.dist_matrix[current_node, next_node]
                            used_time = self.dist_matrix[current_node, next_node] / speed
                            current_time += used_time
                        else:
                            v[3] += speed * left_time
                            break
                    else: # keep servicingonline_customer_data
                        v[4] += time_interval
                else:   # on the road now
                    current_node = v[1][-2]
                    next_node = v[1][-1]
                    speed = self.dist_matrix[current_node, next_node] / self.online_travel_time_data[current_node, next_node]
                    if v[3] + speed * time_interval > self.dist_matrix[current_node, next_node]:   # finish travel and start service
                        # travel
                        used_time = (self.dist_matrix[current_node, next_node] - v[3]) / speed
                        current_time += used_time
                        v[3] = 0
                        print(f"Travel time: {self.online_travel_time_data[current_node, next_node]}")
                        print(f"Arrive to the {next_node} at {current_time}")

                        # service
                        current_node = v[1][-1]
                        left_time = next_time - current_time
                        if left_time > self.online_customer_data[current_node, 6]: # have enough time to finish the service
                            v[4] = self.online_customer_data[current_node, 6]
                            current_time += self.online_customer_data[current_node, 6]
                        else:
                            v[4] += left_time
                            break
                    else: # keep traveling
                        v[3] += speed * time_interval
                        break               

        self.clock = event[0]
        if event[1] == -1:
            self.sample_travel_time()
        print(self.online_customer_data)
        print(self.online_vehicle_data)


    def get_reward(self):
        pass

    def check_terminate(self):
        """
        Check if all customers arrived and were served.
        """
        
        revealed_rows = self.online_customer_data[self.online_customer_data[:, 0] > 0, :]
        is_all_reavled = True if len(revealed_rows) == self.customer_num else False
        is_finished = np.all(revealed_rows[:, 8] == -1)
        
        return is_all_reavled and is_finished

    def reset(self):
        """
        Initialize the data of customer, vehicle and travel_time.
        Return the state_0 for the route-building.
        """

        # initialize data
        self.online_customer_data = self.init_dynamic_customer()
        self.online_vehicle_data = self.init_vehicle_data()
        self.online_travel_time_data = self.sample_travel_time()
        self.dynamic_event_list = self.set_dynamic_event()

        state = self.get_state()

        return state

    def step(self, action):
        if action[0] == True:   # have new routes
            self.set_vehicle_data(action[1])
        
        self.advance_time()
        next_state = self.get_state()
        reward = self.get_reward()
        is_terminate = self.check_terminate()

        return next_state, reward, is_terminate
        