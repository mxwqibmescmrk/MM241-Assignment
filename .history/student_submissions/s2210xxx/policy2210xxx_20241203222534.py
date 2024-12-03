from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2210xxx(Policy):
    def __init__(self):
        self.columns = []  # List to store columns
        self.master_problem = None  # Placeholder for the master problem
        self.subproblem = None  # Placeholder for the subproblem
        self.demands = []  # List to store demands
        self.size = []  # List to store widths of the pieces
        self.W = 50  # Initialize W with a default value
        self.last_stock_idx = 0

    def get_action(self, observation, info):
        # Extract problem parameters from observation and info
        self.demands = [prod["quantity"] for prod in observation["products"]]
        self.size = [prod["size"] for prod in observation["products"]]
       
        self.lengths = [size[0] for size in self.size]
        self.widths = [size[1] for size in self.size]
        self.orders = self.demands

        # Initialize the master problem
        self.initialize_master_problem()

        # Column generation loop
        # while True:
        #     # Step 1: Solve the master problem
        #     solution, dual_values = self.solve_master_problem()
            
        #     # Step 2: Solve the subproblem to find new patterns
        #     new_pattern = self.solve_knapsack(dual_values)

        #     # Step 3: Check if the new pattern improves the solution
        #     if not self.is_improving_pattern(new_pattern, dual_values):
        #         break
            
        #     # Step 4: Add the new pattern to the list of patterns
        #     self.columns.append(new_pattern)
        
        # Return an action based on the solution
        action = self.generate_action(observation)
        return action

    def initialize_master_problem(self):
        # Generate initial cutting patterns (e.g., one pattern per order)
        for i in range(len(self.orders)):
            pattern = [0] * len(self.orders)
            pattern[i] = 1
            self.columns.append(pattern)

    def solve_master_problem(self):
        # Ensure all columns have the same length
        max_length = max(len(col) for col in self.columns)
        for col in self.columns:
            while len(col) < max_length:
                col.append(0)

        # Solve the master problem using the simplex method
        c = [1] * len(self.columns)
        A_eq = np.array(self.columns, dtype=float).T
        b_eq = np.array(self.orders, dtype=float).flatten()
        bounds = [(0, None)] * len(self.columns)
        
        # Ensure b_eq is a 1-D array
        if b_eq.ndim != 1:
            b_eq = b_eq.flatten()
        
        # Adjust dimensions of A_eq and b_eq to match
        if A_eq.shape[0] < b_eq.shape[0]:
            # Pad A_eq with zeros
            padding = np.zeros((b_eq.shape[0] - A_eq.shape[0], A_eq.shape[1]))
            A_eq = np.vstack([A_eq, padding])
        elif A_eq.shape[0] > b_eq.shape[0]:
            # Truncate A_eq
            A_eq = A_eq[:b_eq.shape[0], :]
            
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        solution = result.x
        dual_values = result.slack  # Use slack values as dual values
        return solution, dual_values

    def solve_knapsack(self, dual_values):
        # Solve the knapsack problem to generate a new pattern
        values = dual_values
        weights = self.widths
        capacity = self.W
        n = len(values)
        dp = np.zeros((n + 1, capacity + 1))
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        new_pattern = [0] * n
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                new_pattern[i - 1] = 1
                w -= weights[i - 1]
        
        return new_pattern

    def is_improving_pattern(self, pattern, dual_values):
        # Check if the new pattern improves the solution
        reduced_cost = sum(dual_values[i] * pattern[i] for i in range(len(pattern))) - 1
        return reduced_cost > 0

    def generate_action(self, observation):
        # Sort products by size in descending order
        sorted_products = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        best_action = None
        best_filled_ratio = 0

        for stock_idx in range(self.last_stock_idx, len(observation["stocks"])):
            stock = observation["stocks"][stock_idx]            
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Try placing the product without rotation
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                filled_ratio = self._calculate_filled_ratio(stock, (x, y), prod_size)
                                if filled_ratio > best_filled_ratio:
                                    # list_prods = observation["products"]
                                    total_remaining_products = sum(prod["quantity"] for prod in list_prods if prod["quantity"] > 0)
                                    # print(f"Tổng số sản phẩm còn lại: {total_remaining_products}")
                                    best_filled_ratio = filled_ratio
                                    self.last_stock_idx = stock_idx
                                    if total_remaining_products == 1:
                                        self.last_stock_idx = 0
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
        return {
            "stock_idx": -1,
            "size": self.size[0],
            "position": (0, 0)
        }


    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False
        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

    def _calculate_filled_ratio(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        filled_area = np.sum(stock != -1)
        new_filled_area = filled_area + prod_w * prod_h
        total_area = stock.shape[0] * stock.shape[1]
        return new_filled_area / total_area