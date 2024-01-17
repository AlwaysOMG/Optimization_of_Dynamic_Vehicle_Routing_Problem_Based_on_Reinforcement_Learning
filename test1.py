import numpy as np
from environment.dvrp import DVRP
from iterated_local_search.insertion_heuristic import insertion

file_path = "./instance/Vrp-Set-Solomon/C101.txt"
prob = DVRP(file_path, 0.1, 20)
prob.reset()
capacity = prob.get_vehicle_capacity()
c_data = prob.get_customer_data()
t_data = prob.get_travel_time_data()
a = insertion(c_data, t_data, capacity)
print(c_data)
print(a)