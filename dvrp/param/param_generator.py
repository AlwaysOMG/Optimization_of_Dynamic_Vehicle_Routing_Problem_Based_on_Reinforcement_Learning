import configparser
import random

from dvrp.param.param_class import VehicleParam, NodeParam

config = configparser.ConfigParser()
config.read("./dvrp/dvrp.cfg")
instance_config = config['instance']

class ParamGenerator:
    def __init__(self):
        self.vehicle_num = int(instance_config['vehicle_num'])
        self.customer_num = int(instance_config['customer_num'])
        self.map_size = int(instance_config['map_size'])
        self.customer_demand_lower_limit = int(instance_config['customer_demand_lower_limit'])
        self.customer_demand_upper_limit = int(instance_config['customer_demand_upper_limit'])
        self.vehicle_capacity = int(instance_config['vehicle_capacity'])
        self.time_window_lower_limit = int(instance_config['time_window_lower_limit'])
        self.time_window_upper_limit = int(instance_config['time_window_upper_limit'])
        self.early_penalty_lower_limit = float(instance_config['early_penalty_lower_limit'])
        self.early_penalty_upper_limit = float(instance_config['early_penalty_upper_limit'])
        self.late_penalty_lower_limit = float(instance_config['late_penalty_lower_limit'])
        self.late_penalty_upper_limit = float(instance_config['late_penalty_upper_limit'])

    def generate_parameter(self):
        total_demand = 0
        node_param_list = []
        for i in range(self.customer_num+1):
            x_loc = round(random.uniform(0, self.map_size), 3)
            y_loc = round(random.uniform(0, self.map_size), 3)
            
            if i == 0:
                p = NodeParam(i, x_loc, y_loc)
            else:
                demand = random.randint(self.customer_demand_lower_limit, self.customer_demand_upper_limit)
                earliest_service_time = random.randint(self.time_window_lower_limit, self.time_window_upper_limit-1)
                latest_service_time = random.randint(earliest_service_time+1, self.time_window_upper_limit)
                early_penalty = round(random.uniform(self.early_penalty_lower_limit, self.early_penalty_upper_limit), 3)
                late_penalty = round(random.uniform(self.late_penalty_lower_limit, self.late_penalty_upper_limit), 3)
                p = NodeParam(i, x_loc, y_loc, demand, 
                              earliest_service_time, latest_service_time,
                              early_penalty, late_penalty)
                total_demand += demand
            
            node_param_list.append(p)
        
        vehicle_param_list = []
        for i in range(self.vehicle_num):
            p = VehicleParam(i, self.vehicle_capacity)
            vehicle_param_list.append(p)

        param_list = [node_param_list, vehicle_param_list]
        # check feasibility 
        if total_demand > self.vehicle_capacity * self.vehicle_num:
            param_list = self.generate_parameter()

        return param_list
    

