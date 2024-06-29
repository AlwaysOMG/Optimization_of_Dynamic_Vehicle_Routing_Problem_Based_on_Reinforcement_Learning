class Node:
    def __init__(self, dvrp, param):
        self.dvrp = dvrp
        self.id = param.id
        self.x_loc = param.x_loc
        self.y_loc = param.y_loc
    
    def get_id(self):
        return self.id
    
    def get_obs(self):
        raise NotImplementedError("Subclasses must implement this method")

class Depot(Node):
    def __init__(self, dvrp, param):
        super().__init__(dvrp, param)

    def get_obs(self):       
        return [self.x_loc, self.y_loc, 0, 0, 0, 0, 0, False]

class Customer(Node):
    is_served = False
    service_status = None

    def __init__(self, dvrp, param):
        super().__init__(dvrp, param)
        self.demand = param.demand
        self.earliest_service_time = param.earliest_service_time
        self.latest_service_time = param.latest_service_time
        self.early_penalty = param.early_penalty
        self.late_penalty = param.late_penalty
    
    def receive_service(self, arrive_time):
        self.is_served = True

        if arrive_time < self.earliest_service_time:
            self.service_status = -1
            return self.early_penalty * (self.earliest_service_time - arrive_time)
        elif arrive_time > self.latest_service_time:
            self.service_status = 1
            return self.late_penalty * (arrive_time - self.latest_service_time)
        else:
            self.service_status = 0
            return 0

    def check_served(self):
        return self.is_served
    
    def get_demand(self):
        return self.demand
    
    def get_obs(self):
        return [self.x_loc, self.y_loc, self.demand, 
                self.earliest_service_time, self.latest_service_time, 
                self.early_penalty, self.late_penalty, self.is_served]
    
    def get_service_status(self):
        return self.service_status