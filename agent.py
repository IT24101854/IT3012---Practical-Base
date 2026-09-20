# agent.py

import heapq
import math


class SearchAgent:
    """Search agent that uses A* to navigate toward food."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

        # Current search algorithm
        self.active_algo = 'AStar'

        # Stores the current A* path
        self.plan = []


    # =========================================================
    # STEP 1.1 - HEURISTIC FUNCTIONS
    # =========================================================

    def manhattan_distance(self, pos, goal):
        """
        Manhattan Distance:
        h(n) = |x1 - x2| + |y1 - y2|
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


    def euclidean_distance(self, pos, goal):
        """
        Euclidean Distance:
        h(n) = sqrt((x1-x2)^2 + (y1-y2)^2)
        """
        return math.sqrt(
            (pos[0] - goal[0]) ** 2 +
            (pos[1] - goal[1]) ** 2
        )


    # =========================================================
    # STEP 1.2 - A* SEARCH
    # =========================================================

    def astar_search(
            self,
            start_pos,
            goal_pos,
            walls,
            grid_size,
            heuristic_type='manhattan'
    ):

        # Priority queue
        frontier = []

        # States already explored
        reached_states = set()

        # -----------------------------------------------------
        # Calculate starting heuristic
        # -----------------------------------------------------

        if heuristic_type == 'euclidean':
            h_start = self.euclidean_distance(
                start_pos,
                goal_pos
            )
        else:
            h_start = self.manhattan_distance(
                start_pos,
                goal_pos
            )

        # Starting path cost
        g_start = 0

        # f(n) = g(n) + h(n)
        f_start = g_start + h_start

        # Tuple:
        # (f_cost, g_cost, current_position, path_taken)
        heapq.heappush(
            frontier,
            (f_start, g_start, start_pos, [])
        )

        # -----------------------------------------------------
        # Main A* loop
        # -----------------------------------------------------

        while frontier:

            f_cost, g_cost, current_pos, path_taken = heapq.heappop(
                frontier
            )

            # Goal reached
            if current_pos == goal_pos:
                return path_taken

            # Skip already explored positions
            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            x, y = current_pos

            # -------------------------------------------------
            # Four possible movements
            #
            # The visual game uses:
            # Up    = y + 1
            # Down  = y - 1
            # Left  = x - 1
            # Right = x + 1
            # -------------------------------------------------

            neighbors = [
                ((x, y + 1), 'Up'),
                ((x, y - 1), 'Down'),
                ((x - 1, y), 'Left'),
                ((x + 1, y), 'Right')
            ]

            for neighbor_pos, action in neighbors:

                nx, ny = neighbor_pos

                # Check grid boundaries
                within_bounds = (
                        0 <= nx < grid_size[0]
                        and
                        0 <= ny < grid_size[1]
                )

                # Only explore valid cells
                if (
                        within_bounds
                        and neighbor_pos not in walls
                        and neighbor_pos not in reached_states
                ):

                    # Every movement costs 1
                    g_new = g_cost + 1

                    # Calculate heuristic
                    if heuristic_type == 'euclidean':
                        h_new = self.euclidean_distance(
                            neighbor_pos,
                            goal_pos
                        )
                    else:
                        h_new = self.manhattan_distance(
                            neighbor_pos,
                            goal_pos
                        )

                    # Calculate f(n)
                    f_new = g_new + h_new

                    # Add action to path
                    new_path = path_taken + [action]

                    # Add neighbor to priority queue
                    heapq.heappush(
                        frontier,
                        (
                            f_new,
                            g_new,
                            neighbor_pos,
                            new_path
                        )
                    )

        # No path found
        return None


    # =========================================================
    # STEP 1.3 - AGENT DECISION LOOP
    # =========================================================

    def sense_and_act(self, percept: dict):

        current_pos = percept['agent_pos']

        # -----------------------------------------------------
        # If we already have a path, continue following it
        # -----------------------------------------------------

        if self.plan:
            return self.plan.pop(0)

        # -----------------------------------------------------
        # A* SEARCH
        # -----------------------------------------------------

        if self.active_algo == 'AStar':

            remaining_food = percept['remaining_food']

            # No food remaining
            if not remaining_food:
                return None

            # -------------------------------------------------
            # Find the closest food using Manhattan distance
            # -------------------------------------------------

            goal_pos = min(
                remaining_food,
                key=lambda food: self.manhattan_distance(
                    current_pos,
                    food
                )
            )

            # -------------------------------------------------
            # Get environment information
            # -------------------------------------------------

            walls = percept['walls']
            traps = percept['traps']
            grid_size = percept['grid_size']

            # -------------------------------------------------
            # Treat walls AND purple traps as blocked
            # -------------------------------------------------

            blocked_cells = walls | traps

            # -------------------------------------------------
            # Find path using A*
            # -------------------------------------------------

            self.plan = self.astar_search(
                current_pos,
                goal_pos,
                blocked_cells,
                grid_size,
                heuristic_type='manhattan'
            )

            # Return first action
            if self.plan:
                return self.plan.pop(0)

        # IMPORTANT:
        # Do NOT make a random move if no safe path exists.
        return None


# =============================================================
# TEST STEP 1.1 AND STEP 1.2
# =============================================================

if __name__ == "__main__":

    agent = SearchAgent()

    start = (0, 0)
    goal = (3, 4)

    print(
        "Manhattan Distance:",
        agent.manhattan_distance(start, goal)
    )

    print(
        "Euclidean Distance:",
        agent.euclidean_distance(start, goal)
    )

    walls = set()
    grid_size = (5, 5)

    path = agent.astar_search(
        start,
        goal,
        walls,
        grid_size,
        heuristic_type='manhattan'
    )

    print("A* Path:", path)