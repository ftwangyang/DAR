import os
import time
import math

class GreedyCVRPSolver:
    def __init__(self, vrplib_path, repeat_times=1):
        self.vrplib_path = vrplib_path
        self.repeat_times = repeat_times

    def calculate_distance(self, node1, node2):
        return math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)

    def read_vrp_instance(self, filepath):
        nodes, demands, depot, capacity = [], [], None, None
        reading_nodes, reading_demands = False, False
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('CAPACITY'):
                    capacity = int(line.split()[-1])
                elif line.startswith('NODE_COORD_SECTION'):
                    reading_nodes = True
                    continue
                elif line.startswith('DEMAND_SECTION'):
                    reading_nodes = False
                    reading_demands = True
                    continue
                elif line.startswith('DEPOT_SECTION'):
                    depot = int(file.readline().strip())
                    break

                if reading_nodes:
                    parts = line.split()
                    nodes.append({'id': int(parts[0]), 'x': float(parts[1]), 'y': float(parts[2])})
                elif reading_demands:
                    parts = line.split()
                    demands.append(int(parts[1]))

        for i, node in enumerate(nodes):
            node['demand'] = demands[i]
        return nodes, depot, capacity

    def greedy_solve(self, nodes, depot, capacity):
        visited, path, current_load = set(), [], 0
        depot_node = nodes[depot - 1]
        current_node = depot_node
        visited.add(depot)

        while len(visited) < len(nodes):
            nearest_node, nearest_distance = None, float('inf')

            for node in nodes:
                if node['id'] not in visited and current_load + node['demand'] <= capacity:
                    distance = self.calculate_distance(current_node, node)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_node = node

            if nearest_node is None:
                current_load = 0
                current_node = depot_node
                if path[-1] != depot:
                    path.append(depot)
            else:
                visited.add(nearest_node['id'])
                path.append(nearest_node['id'])
                current_load += nearest_node['demand']
                current_node = nearest_node

        if path[-1] != depot:
            path.append(depot)
        return path

    def calculate_total_distance(self, path, nodes):
        total_distance = 0
        for i in range(1, len(path)):
            total_distance += self.calculate_distance(nodes[path[i - 1] - 1], nodes[path[i] - 1])
        return total_distance

    def test_on_one_ins(self, instance_file):
        nodes, depot, capacity = self.read_vrp_instance(instance_file)
        path = self.greedy_solve(nodes, depot, capacity)
        total_distance = self.calculate_total_distance(path, nodes)
        print(f"Total distance for {os.path.basename(instance_file)}: {total_distance}")
        return path, total_distance

    def test_on_vrplib(self):
        files = [f for f in os.listdir(self.vrplib_path) if f.endswith('.vrp')]
        for t in range(self.repeat_times):
            for filename in files:
                instance_file = os.path.join(self.vrplib_path, filename)
                print(f"Testing instance: {filename}")
                start_time = time.time()
                path, total_distance = self.test_on_one_ins(instance_file)
                elapsed_time = time.time() - start_time
                print(f"Instance {filename}, Time: {elapsed_time:.2f}s, Path: {path}, Total Distance: {total_distance}")

# test
solver = GreedyCVRPSolver('./DAR/VRPLib/Vrp-Set-X/X', repeat_times=1)
solver.test_on_vrplib()
