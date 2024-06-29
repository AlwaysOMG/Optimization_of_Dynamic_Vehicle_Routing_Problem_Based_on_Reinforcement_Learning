class VehicleParam:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity

class NodeParam:
    def __init__(self, id, x_loc, y_loc, demand = None,
                 earliest_service_time = None, latest_service_time = None,
                 early_penalty = None, late_penalty = None):
        self.id = id
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.demand = demand
        self.earliest_service_time = earliest_service_time
        self.latest_service_time = latest_service_time
        self.early_penalty = early_penalty
        self.late_penalty = late_penalty        