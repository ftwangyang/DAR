import os
import time
import math

class TSPSolver:
    def __init__(self, tsplib_path, repeat_times=1):
        self.tsplib_path = tsplib_path
        self.repeat_times = repeat_times
        self.total_distances = []

    def calculate_distance(self, node1, node2):
        return math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)

    def read_tsp_instance(self, filepath):
        nodes = []
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('DIMENSION'):
                    num_nodes = int(line.split()[-1])
                elif line.startswith('EDGE_WEIGHT_TYPE'):
                    edge_weight_type = line.split()[-1]
                elif line.startswith('NODE_COORD_SECTION'):
                    for i in range(num_nodes):
                        parts = file.readline().split()
                        nodes.append({'id': int(parts[0]), 'x': float(parts[1]), 'y': float(parts[2])})
                elif line.startswith('DEPOT_SECTION'):
                    break

        return nodes

    def greedy_solve(self, nodes):
        visited, path = set(), []
        current_node = nodes[0]
        visited.add(0)

        while len(visited) < len(nodes):
            nearest_node, nearest_distance = None, float('inf')

            for node in nodes:
                if node['id'] not in visited:
                    distance = self.calculate_distance(current_node, node)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_node = node

            visited.add(nearest_node['id'])
            path.append(nearest_node['id'])
            current_node = nearest_node

        path.append(nodes[0]['id'])
        return path

    def calculate_total_distance(self, path, nodes):
        total_distance = 0
        for i in range(1, len(path)):
            total_distance += self.calculate_distance(nodes[path[i - 1] - 1], nodes[path[i] - 1])
        return total_distance

    def test_on_one_ins(self, instance_file):
        nodes = self.read_tsp_instance(instance_file)
        path = self.greedy_solve(nodes)
        total_distance = self.calculate_total_distance(path, nodes)
        self.total_distances.append(total_distance)
        print(f"Total distance for {os.path.basename(instance_file)}: {total_distance}")
        return path, total_distance

    def test_on_tsplib(self):
        files = [f for f in os.listdir(self.tsplib_path) if f.endswith('.tsp')]
        for t in range(self.repeat_times):
            for filename in files:
                instance_file = os.path.join(self.tsplib_path, filename)
                print(f"Testing instance: {filename}")
                start_time = time.time()
                path, total_distance = self.test_on_one_ins(instance_file)
                elapsed_time = time.time() - start_time
                print(f"Instance {filename}, Time: {elapsed_time:.2f}s, Path: {path}, Total Distance: {total_distance}")

        avg_total_distance = sum(self.total_distances) / len(self.total_distances)
        print(f"Average total distance for all instances: {avg_total_distance}")

# Test
solver = TSPSolver('./DAR/TSP/TSPLib1', repeat_times=1)
solver.test_on_tsplib()