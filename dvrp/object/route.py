class Route:
    def __init__(self):
        self.route_list = []
    
    def add_node(self, node):
        self.route_list.append(node)       
    
    def get_next_customer(self):
        return self.route_list.pop(0) if self.route_list else None
    
    def display(self):
        l = [node.get_id() for node in self.route_list]
        print(l)