OVERTIME_PENALTY = 1000
CAPACITY_PENALTY = 999999999
INF = CAPACITY_PENALTY + 1

def cal_cost(route, c_data, t_data, capacity):
    # check capacity
    demand = 0
    for customer in route:
        demand += c_data[customer, 3]
    if demand > capacity:
        return CAPACITY_PENALTY

    cost = 0
    current_time = 0
    current_node = 0
    for customer in route:
        current_time += t_data[current_node, customer]
        if current_time < c_data[customer, 4]:
            # wait until the earliest service time
            current_time = c_data[customer, 4]
        elif current_time > c_data[customer, 5]:
            # overtime
            cost += (current_time - c_data[customer, 5]) * OVERTIME_PENALTY
        current_time += c_data[customer, 6]
        current_node = customer

    cost += current_time
    return cost

def or_opt(routes, c_data, t_data):
    for idx, r in enumerate(routes):
        best = r
        best_cost = cal_cost(r, c_data, t_data)
        customer_num = len(r)-2
        for l in range(1, customer_num):
            for i in range(1, customer_num-l+1):
                for j in range(1, customer_num-l+1):
                    if i == j:
                        continue
                    
                    new = r[:]
                    cut = r[i:i+l]
                    del new[i:i+l]
                    new[j:j] = cut

                    new_cost = cal_cost(new, c_data, t_data)
                    if new_cost < best_cost:
                        best = new
                        best_cost = new_cost
                        print(best)

        routes[idx] = best

def two_opt_star(routes, c_data, t_data, capacity):
    current_total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)
    best_total_cost = INF
    while(current_total_cost < best_total_cost):
        best_total_cost = current_total_cost

        # pick two routes
        for r_i in range(len(routes)):
            for r_j in range(r_i+1, len(routes)):
                current_route_i = routes[r_i][:]
                current_route_j = routes[r_j][:]
                best_route_i = current_route_i
                best_route_j = current_route_j
                best_cost = cal_cost(best_route_i, c_data, t_data, capacity) + \
                    cal_cost(best_route_j, c_data, t_data, capacity)

                # pick two customers
                for i in range(1, len(routes[r_i])):
                    for j in range(1, len(routes[r_j])):
                        new_i_a = current_route_i[:i]
                        new_i_b = current_route_i[i:]
                        new_j_a = current_route_j[:j]
                        new_j_b = current_route_j[j:]
                        new_i_a.extend(new_j_b)
                        new_j_a.extend(new_i_b)
                        
                        new_cost = cal_cost(new_i_a, c_data, t_data, capacity) + \
                            cal_cost(new_j_a, c_data, t_data, capacity)
                        if new_cost < best_cost:
                            best_route_i = new_i_a
                            best_route_j = new_j_a
                            best_cost = new_cost
                
                # change the route
                routes[r_i] = best_route_i
                routes[r_j] = best_route_j
        
        # clear empty route
        while([0, 0] in routes):
            routes.remove([0, 0])
        
        current_total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)

def exchange(routes, c_data, t_data, capacity):
    current_total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)
    best_total_cost = INF
    while(current_total_cost < best_total_cost):
        best_total_cost = current_total_cost

        # pick two routes
        for r_i in range(len(routes)):
            for r_j in range(len(routes)):
                if r_i == r_j and len(routes[r_i]) > 3: # intra-route exchange
                    current_route = routes[r_i][:]
                    best_route = current_route
                    best_cost = cal_cost(best_route, c_data, t_data, capacity)
                    
                    # pick two customers
                    for i in range(1, len(routes[r_i])-1):
                        for j in range(i+1, len(routes[r_i])-1):                            
                            new = current_route[:]                            
                            new[i] = current_route[j]
                            new[j] = current_route[i]

                            new_cost = cal_cost(new, c_data, t_data, capacity)
                            if new_cost < best_cost:
                                best_route = new
                                best_cost = new_cost
                    
                    # optimize the route
                    routes[r_i] = best_route
                elif r_i != r_j:    # inter-route exchange
                    current_route_i = routes[r_i][:]
                    current_route_j = routes[r_j][:]
                    best_route_i = current_route_i[:]
                    best_route_j = current_route_j[:]
                    best_cost = cal_cost(best_route_i, c_data, t_data, capacity) + \
                        cal_cost(best_route_j, c_data, t_data, capacity)
                    
                    # pick two customers
                    for i in range(1, len(routes[r_i])-1):
                        for j in range(1, len(routes[r_j])-1):
                            new_i = current_route_i[:]
                            new_j = current_route_j[:]
                            new_i[i] = current_route_j[j]
                            new_j[j] = current_route_i[i]
                            
                            new_cost = cal_cost(new_i, c_data, t_data, capacity) + \
                                cal_cost(new_j, c_data, t_data, capacity)
                            if new_cost < best_cost:
                                best_route_i = new_i
                                best_route_j = new_j
                                best_cost = new_cost
                
                    # optimize the route
                    routes[r_i] = best_route_i
                    routes[r_j] = best_route_j
        
        # clear empty route
        while([0, 0] in routes):
            routes.remove([0, 0])

        current_total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)

def relocate(routes, c_data, t_data, capacity):
    current_total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)
    best_total_cost = INF
    while(current_total_cost < best_total_cost):
        best_total_cost = current_total_cost

        # pick two routes
        for r_i in range(len(routes)):
            for r_j in range(len(routes)):
                if r_i == r_j and len(routes[r_i]) > 3: # intra-route relocate
                    current_route = routes[r_i][:]
                    best_route = current_route[:]
                    best_cost = cal_cost(best_route, c_data, t_data, capacity)

                    # pick one customer
                    for i in range(1, len(current_route)-1):
                        new = current_route[:]
                        del new[i]                        
                        for pos in range(1, len(new)):
                            if i == pos:
                                continue
                            else:
                                new.insert(pos, current_route[i])

                            new_cost = cal_cost(new, c_data, t_data, capacity)
                            if new_cost < best_cost:
                                best_route = new
                                best_cost = new_cost
                    
                    # optimize the route
                    routes[r_i] = best_route
                elif r_i != r_j:        # inter-route relocate
                    current_route_i = routes[r_i][:]
                    current_route_j = routes[r_j][:]
                    best_route_i = current_route_i[:]
                    best_route_j = current_route_j[:]
                    best_cost = cal_cost(best_route_i, c_data, t_data, capacity) + \
                        cal_cost(best_route_j, c_data, t_data, capacity)
                    
                    # pick one customer
                    for i in range(1, len(current_route_i)-1):
                        for pos in range(1, len(current_route_j)):
                            new_i = current_route_i[:]
                            new_j = current_route_j[:]
                            del new_i[i]
                            new_j.insert(pos, current_route_i[i])
                            
                            new_cost = cal_cost(new_i, c_data, t_data, capacity) + \
                                cal_cost(new_j, c_data, t_data, capacity)
                            if new_cost < best_cost:
                                best_route_i = new_i
                                best_route_j = new_j
                                best_cost = new_cost
                
                    routes[r_i] = best_route_i
                    routes[r_j] = best_route_j
            
        while([0, 0] in routes):
            routes.remove([0, 0])

        current_total_cost = sum(cal_cost(r, c_data, t_data, capacity) for r in routes)
