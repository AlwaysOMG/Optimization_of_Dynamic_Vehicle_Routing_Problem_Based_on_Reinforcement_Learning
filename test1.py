from model.dvrp import Env
from opt_method.utils import cal_cost
from opt_method.iterated_local_search import iterated_local_search

file_path = "./instance/Vrp-Set-Solomon/R101.txt"
prob = Env(file_path, 0, 20)
prob.reset()
capacity = prob.get_vehicle_capacity()
c_data = prob.get_customer_data()
t_data = prob.get_travel_time_data()

routes = iterated_local_search(c_data, t_data, capacity)
cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)
print(routes)
print(cost)
