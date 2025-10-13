import numpy as np
import heapq
import itertools
from collections import deque

class Agent:
    def __init__(self, start_pos, destination, grid_world, weight_matrix, altruism_points):
        self.start_position = np.array(start_pos)
        self.position = np.array(start_pos)
        self.destination = np.array(destination)
        self.grid_world = grid_world
        self.weight_matrix = weight_matrix
        self.altruism_points = altruism_points
        self.path_queue = deque()
        self.past_positions = []

    def reset_position(self):
        """Reset the agent's position to its starting position."""
        self.position = self.start_position
        self.path_queue = deque()
        self.past_positions = []

    def get_position(self):
        return self.position
    
    def is_at_destination(self):
        return np.array_equal(self.position, self.destination)

    def move_towards(self, target_pos):
        if not self.path_queue or not np.array_equal(target_pos, self.path_queue[-1]):
            self._calculate_path(target_pos)
            redundant = self.path_queue.popleft()

        if self.path_queue:
            self.position = self.path_queue.popleft()
            for rider in self.riders:
                rider.position = self.position

    def _calculate_path(self, target_pos):
        self.path_queue = deque(self._dijkstra_path(self.position, target_pos))

    def _dijkstra_path(self, start, target):
        height, width = self.grid_world
        visited = set()
        min_heap = [(0, tuple(start), None)]
        parent_map = {}

        # Direction indices in weight_matrix[x, y, direction]
        # 0: Up, 1: Down, 2: Left, 3: Right
        directions = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

        while min_heap:
            cost, current, prev = heapq.heappop(min_heap)
            if current in visited:
                continue
            visited.add(current)
            parent_map[current] = prev

            if np.array_equal(current, target):
                break

            x, y = current
            for dx, dy, dir_idx in directions:
                next_pos = (x + dx, y + dy)
                if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width:
                    # Use directional weight for the specific movement
                    next_cost = cost + self.weight_matrix[x, y, dir_idx]
                    if next_pos not in visited:
                        heapq.heappush(min_heap, (next_cost, next_pos, current))

        path = []
        step = tuple(target)
        while step:
            path.append(step)
            step = parent_map.get(step)

        return list(reversed(path))

    def _calculate_path_length(self, path):
        if len(path) <= 1:
            return 0
            
        total_length = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            
            # Determine direction index
            if x2 - x1 == -1:  # Moving up
                dir_idx = 0
            elif x2 - x1 == 1:  # Moving down
                dir_idx = 1
            elif y2 - y1 == -1:  # Moving left
                dir_idx = 2
            elif y2 - y1 == 1:  # Moving right
                dir_idx = 3
            
            total_length += self.weight_matrix[x1, y1, dir_idx]
        
        return total_length

class Driver(Agent):
    def __init__(self, max_capacity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_capacity = max_capacity
        self.riders = []

    def select_rider(self, rider):
        score = rider.altruism_points - self._calculate_detour_cost(rider)
        return score

    def add_rider(self, rider):
        if len(self.riders) < self.max_capacity:
            self.riders.append(rider)
            self.altruism_points += rider.altruism_points
            rider.being_picked_up = True
            return True
        return False
    
    def _calculate_positive_reward(self, rider):
        start = tuple(rider.get_position())
        destination = tuple(rider.destination)

        shortest_path = self._dijkstra_path(start, destination)
        shortest_distance = self._calculate_path_length(shortest_path)

        return shortest_distance

    def _calculate_detour_cost(self, new_rider):
        """Calculate the true detour cost of adding a new rider to existing riders."""
        import itertools
        
        current_pos = tuple(self.get_position())
        
        direct_path = self._dijkstra_path(current_pos, tuple(self.destination))
        direct_distance = self._calculate_path_length(direct_path)
        
        if not self.riders:
            
            pickup_pos = tuple(new_rider.get_position())
            dropoff_pos = tuple(new_rider.destination)
            
            path_to_pickup = self._dijkstra_path(current_pos, pickup_pos)
            path_to_dropoff = self._dijkstra_path(pickup_pos, dropoff_pos)
            path_to_destination = self._dijkstra_path(dropoff_pos, tuple(self.destination))
            
            detour_distance = (
                self._calculate_path_length(path_to_pickup)
                + self._calculate_path_length(path_to_dropoff)
                + self._calculate_path_length(path_to_destination)
            )
            
            detour_cost = detour_distance - direct_distance
        else:
            # Calculate optimal path with existing riders
            existing_destinations = [tuple(rider.destination) for rider in self.riders]
            
            # Find optimal route with current riders
            min_current_distance = float('inf')
            for perm in itertools.permutations(range(len(existing_destinations))):
                path_length = 0
                pos = current_pos
                
                for idx in perm:
                    dest = existing_destinations[idx]
                    path = self._dijkstra_path(pos, dest)
                    path_length += self._calculate_path_length(path)
                    pos = dest
                
                # Add final leg to driver's destination
                final_path = self._dijkstra_path(pos, tuple(self.destination))
                path_length += self._calculate_path_length(final_path)
                
                min_current_distance = min(min_current_distance, path_length)
            
            # Calculate optimal path including the new rider
            new_pickup = tuple(new_rider.get_position())
            new_dropoff = tuple(new_rider.destination)
            all_locations = existing_destinations + [new_dropoff]
            
            # We need to insert the pickup first, then consider all permutations of dropoffs
            min_new_distance = float('inf')
            
            # First, go to pickup location
            pickup_path = self._dijkstra_path(current_pos, new_pickup)
            pickup_distance = self._calculate_path_length(pickup_path)
            
            # Then consider all permutations of dropoffs including the new rider
            for perm in itertools.permutations(range(len(all_locations))):
                path_length = pickup_distance
                pos = new_pickup
                
                for idx in perm:
                    dest = all_locations[idx]
                    path = self._dijkstra_path(pos, dest)
                    path_length += self._calculate_path_length(path)
                    pos = dest
                
                # Add final leg to driver's destination
                final_path = self._dijkstra_path(pos, tuple(self.destination))
                path_length += self._calculate_path_length(final_path)
                
                min_new_distance = min(min_new_distance, path_length)
            
            # True detour cost is the difference
            detour_cost = min_new_distance - min_current_distance
        
        # Apply same penalty as before if applicable
        if direct_distance - detour_cost > 6:
            detour_cost *= 1.5
            
        return detour_cost

class Rider(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.being_picked_up = False

    def request_ride(self, driver):
        if self.altruism_points > 0 and not self.being_picked_up:
            if driver.add_passenger(self):
                self.altruism_points -= 1
                return True
        return False

def create_agent(role, start_pos, destination, grid_world, weight_matrix, altruism_points, max_capacity=None):
    if role == 'driver':
        return Driver(start_pos=start_pos, destination=destination, grid_world=grid_world, weight_matrix=weight_matrix, altruism_points=altruism_points, max_capacity=max_capacity)
    else:
        return Rider(start_pos=start_pos, destination=destination, grid_world=grid_world, weight_matrix=weight_matrix, altruism_points=altruism_points)

def switch_role(agent, new_role, max_capacity=None):
    """
    Switch the role of an agent. If the role remains the same, reset its position.
    """
    if new_role == 'driver' and isinstance(agent, Rider):
        new_agent = Driver(max_capacity, agent.start_position, agent.destination, agent.grid_world, agent.weight_matrix, agent.altruism_points)
    elif new_role == 'rider' and isinstance(agent, Driver):
        new_agent = Rider(agent.start_position, agent.destination, agent.grid_world, agent.weight_matrix, agent.altruism_points)
    else:
        # If the role remains the same, reset the agent's position
        agent.reset_position()
        return agent  # No change if already in the correct role

    return new_agent