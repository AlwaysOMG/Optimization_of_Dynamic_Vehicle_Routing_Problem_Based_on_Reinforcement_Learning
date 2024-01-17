import heapq

class Event:
    def __init__(self, time):
        self.time = time

    def __lt__(self, other):
        return self.time < other.time
    
    def process(self):
        pass

class ArrivalEvent(Event):
    def __init__(self, time, customer_id):
        super().__init__(time)
        self.customer_id = customer_id

    def process(self):
        print(f"{self.time}: customer {self.customer_id} arrival.")

class TravelTimeUpdateEvent(Event):
    def __init__(self, time):
        super().__init__(time)

    def process(self):
        print(f"{self.time}: Update travel time.")

class StartServiceEvent(Event):
    def __init__(self, time, vehicle_id, customer_id):
        super().__init__(time)
        self.vehicle_id = vehicle_id
        self.customer_id = customer_id

    def process(self):
        print(f"{self.time}: vehicle {self.vehicle_id} arrives customer {self.customer_id}.")

class FinishServiceEvent(Event):
    def __init__(self, time, vehicle_id, customer_id):
        super().__init__(time)
        self.vehicle_id = vehicle_id
        self.customer_id = customer_id

    def process(self):
        print(f"{self.time}: vehicle {self.vehicle_id} completes customer {self.customer_id}.")

class EventManager:
    def __init__(self):
        self.event_queue = []
        self.current_time = 0

    def add_event(self, event):
        heapq.heappush(self.event_queue, event)

    def process_event(self):
        time, event = heapq.heappop(self.event_queue)
        
        return time, event
