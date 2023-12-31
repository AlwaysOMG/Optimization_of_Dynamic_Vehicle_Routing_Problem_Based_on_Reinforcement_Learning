import numpy as np
import copy
import time

def cal_savings(t_data):
    customer_num = t_data.shape[0]
    savings = np.full((customer_num, customer_num), -1, dtype=float)

    for i in range(1, customer_num):
        for j in range(i+1, customer_num):           
            s = t_data[0, i] + t_data[0, j] - t_data[i, j]
            savings[i, j] = s

    # np.savetxt("savings.txt", savings, fmt='%f', delimiter='\t')

    return savings

def check_capacity(route, c_data, capacity):
    demand = 0
    for customer in route:
        demand += c_data[customer, 3]
    if demand > capacity:
        return False
    return True

def check_time_window(route, c_data, t_data, wait_time):
    t = 0
    previous_node = 0
    for customer in route:
        t += t_data[previous_node, customer]
        
        if t > c_data[0, 5] or t > c_data[customer, 5]:
            return False
        if t + wait_time >= c_data[customer, 4]:
            t = t + c_data[customer, 6] if wait_time == 0 \
            else c_data[customer, 4] + c_data[customer, 6]
            previous_node = customer
        else:
            return False

    return True

def saving_method(c_data, capacity, t_data):
    savings = cal_savings(t_data)
    indices = np.argsort(-savings, axis=None)
    savings_i, savings_j = np.unravel_index(indices, t_data.shape)

    customer_num = len(c_data)-1
    available_customer_num = len(np.where(c_data[:, 7] == 0)[0])-1
    routes = []
    wait_time = 0
    flag = 0
    while(flag < len(savings_i)):
        customer_i, customer_j = savings_i[flag], savings_j[flag]
        if c_data[customer_i, 7] != 0 or c_data[customer_j, 7] != 0:
            flag += 1
            continue

        if savings[customer_i, customer_j] == -1:
            assign_customer_num = sum(len(r) for r in routes)
            if assign_customer_num == available_customer_num:
                break
            elif assign_customer_num > 0.9 * available_customer_num:
                for i in range(1, customer_num):
                    if c_data[i, 7] == 0 and c_data[i, 8] == 0:
                        routes.append([i])
                        route_idx = len(routes) - 1
                        c_data[i, 8] = 1 + route_idx
                break
            else:
                wait_time += 10
                flag = 0
                continue                    

        if c_data[customer_i, 8] == 0 and c_data[customer_j, 8] == 0:
            for case in range(2):
                route = [customer_i, customer_j] if case == 0 else [customer_j, customer_i]                
                if check_capacity(route, c_data, capacity) and check_time_window(route, c_data, t_data, wait_time):
                    routes.append(route)
                    route_idx = len(routes) - 1
                    c_data[customer_i, 8] = 1 + route_idx
                    c_data[customer_j, 8] = 1 + route_idx
                    flag = 0
                    break
        elif c_data[customer_i, 8] > 0 and c_data[customer_j, 8] == 0:
            route_i_idx = c_data[customer_i, 8] - 1
            for pos in range(len(routes[route_i_idx])+1):
                new_route = copy.deepcopy(routes[route_i_idx])
                new_route.insert(pos, customer_j)
                if check_capacity(new_route, c_data, capacity) and check_time_window(new_route, c_data, t_data, wait_time):
                    routes[route_i_idx] = new_route
                    c_data[customer_j, 8] = 1 + route_i_idx
                    flag = 0
                    break
        elif c_data[customer_i, 8] == 0 and c_data[customer_j, 8] > 0:
            route_j_idx = c_data[customer_j, 8] - 1
            for pos in range(len(routes[route_j_idx])+1):
                new_route = copy.deepcopy(routes[route_j_idx])
                new_route.insert(pos, customer_i)
                if check_capacity(new_route, c_data, capacity) and check_time_window(new_route, c_data, t_data, wait_time):
                    routes[route_j_idx] = new_route
                    c_data[customer_i, 8] = 1 + route_j_idx
                    flag = 0
                    break

        flag += 1

    return routes, c_data
