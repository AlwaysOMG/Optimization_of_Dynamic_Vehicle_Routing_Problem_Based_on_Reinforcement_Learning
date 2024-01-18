from model.dvrp import Env
from opt_method.init_sol import insertion
from opt_method.local_search import cal_cost, two_opt_star, exchange, relocate

file_path = "./instance/Vrp-Set-Solomon/R101.txt"
prob = Env(file_path, 0, 20)
prob.reset()
capacity = prob.get_vehicle_capacity()
c_data = prob.get_customer_data()
t_data = prob.get_travel_time_data()

a = insertion(c_data, t_data, capacity)
total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in a)
best_total_cost = 9999999
for q in a:
    print(q)
print(len(a))
print(total_cost)

while(total_cost < best_total_cost):
    best_total_cost = total_cost
    
    exchange(a, c_data, t_data, capacity)
    total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in a)
    for q in a:
        print(q)
    print(len(a))
    print(f"after exchange: {total_cost}")
    print("==========================")

    relocate(a, c_data, t_data, capacity)
    total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in a)
    for q in a:
        print(q)
    print(len(a))
    print(f"after relocate: {total_cost}")
    print("==========================")

    two_opt_star(a, c_data, t_data, capacity)
    total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in a)
    for q in a:
        print(q)
    print(len(a))
    print(f"after 2-opt: {total_cost}")
    print("==========================")

