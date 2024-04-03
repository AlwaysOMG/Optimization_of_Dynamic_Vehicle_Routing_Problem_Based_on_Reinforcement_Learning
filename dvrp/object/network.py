import numpy as np

class Road:
    vehicle = None

    def __init__(self, node_1, node_2, cv):
        self.start_node = node_1
        self.end_node = node_2
        self.dist = ((node_1.x_loc - node_2.x_loc) ** 2 + 
                     (node_1.y_loc - node_2.y_loc) ** 2) ** 0.5
        self.std = cv * self.dist
        self.travel_time = 0
        self.speed = 0
        
        self.sample_travel_time()
    
    def sample_travel_time(self):
        self.travel_time = np.random.normal(self.dist, self.std, 1)[0]
        while self.travel_time <= 0:
            self.travel_time = np.random.normal(self.dist, self.std, 1)[0]
        self.speed = self.dist / self.travel_time
    
    def get_dist(self):
        return self.dist
    
    def get_travel_time(self):
        return self.travel_time

    def get_speed(self):
        return self.speed
    
    def get_destination(self):
        return self.end_node

class Network:
    def __init__(self, node_list, cv):
        self.road_matrix = []
        for node_1 in node_list:
            road_list = []
            for node_2 in node_list:
                if node_1 == node_2:
                    r = None
                else:
                    r = Road(node_1, node_2, cv)
                road_list.append(r)
            self.road_matrix.append(road_list)
    
    def update_travel_time(self):
        for row in self.road_matrix:
            for road in row:
                if road != None:
                    road.sample_travel_time()
    
    def get_road(self, node_1, node_2):
        id_1 = node_1.get_id()
        id_2 = node_2.get_id()
        return self.road_matrix[id_1][id_2]
    