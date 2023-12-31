import numpy as np
from dvrp.dvrp import DVRP
from local_search.init_sol import saving_method

np.set_printoptions(threshold=np.inf)

file_path = "./instance/Vrp-Set-Solomon/R101.txt"
prob = DVRP(file_path, 0.1, 20)
capacity = prob.online_vehicle_data[0][2]
a, prob.online_customer_data = saving_method(prob.online_customer_data, capacity, prob.online_travel_time_data)

print(a)
print(prob.online_customer_data)