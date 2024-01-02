OVERTIME_PENALTY = 1000

def cal_cost(route, c_data, t_data, log=False):
    route.append(0)
    cost = 0
    current_time = 0
    current_node = 0
    for customer in route:
        current_time += t_data[current_node, customer]
        if current_time < c_data[customer, 4]:
            current_time = c_data[customer, 4]
        elif current_time > c_data[customer, 5]:
            overtime = current_time - c_data[customer, 5]
            cost += overtime * OVERTIME_PENALTY
            if log:
                print(f"{customer} / {overtime}")
        current_time += c_data[customer, 6]
        current_node = customer

    cost += current_time
    route.pop()
    return cost

def or_opt(routes, c_data, t_data):
    for idx, r in enumerate(routes):
        best = r
        best_cost = cal_cost(r, c_data, t_data)
        customer_num = len(r)
        for l in range(1, customer_num):
            for i in range(customer_num-l+1):
                for j in range(customer_num-l+1):
                    if i == j:
                        continue
                    
                    new = r[:]
                    cut = new[i:i+l]
                    del new[i:i+l]
                    new[j:j] = cut

                    new_cost = cal_cost(new, c_data, t_data)
                    if new_cost < best_cost:
                        best = new
                        best_cost = new_cost

        routes[idx] = best

def two_opt_star(routes, c_data, t_data):
    current_total_cost = sum(cal_cost(r, c_data, t_data) for r in routes)
    best_total_cost = 999999999
    while(current_total_cost < best_total_cost):
        vehicle_num = len(routes)
        best_total_cost = current_total_cost

        for r_i in range(vehicle_num):
            for r_j in range(r_i+1, vehicle_num):
                best_i = routes[r_i][:]
                best_j = routes[r_j][:]
                best_cost = cal_cost(best_i, c_data, t_data) + cal_cost(best_j, c_data, t_data)
                for i in range(len(routes[r_i])+1):
                    for j in range(len(routes[r_j])+1):
                        cut_i_a = routes[r_i][:i]
                        cut_i_b = routes[r_i][i:]
                        cut_j_a = routes[r_j][:j]
                        cut_j_b = routes[r_j][j:]
                        cut_i_a.extend(cut_j_b)
                        cut_j_a.extend(cut_i_b)
                        new_cost = cal_cost(cut_i_a, c_data, t_data) + cal_cost(cut_j_a, c_data, t_data)

                        if new_cost < best_cost:
                            best_i = cut_i_a
                            best_j = cut_j_a
                            best_cost = new_cost

                routes[r_i] = best_i
                routes[r_j] = best_j
        
        while([] in routes):
            routes.remove([])

def relocate(routes, c_data, t_data):
    current_total_cost = sum(cal_cost(r, c_data, t_data) for r in routes)
    best_total_cost = 999999999
    while(current_total_cost < best_total_cost):
        vehicle_num = len(routes)
        best_total_cost = current_total_cost

        for r_i in range(vehicle_num):
            for r_j in range(r_i, vehicle_num):
                if r_i == r_j:
                    pass
                else:
                    pass

def exchange():
    pass

if __name__ == '__main__':
    route = [1,2,3,4]
