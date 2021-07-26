import numpy as np
import helper
import random
import math

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to balance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    def save_best_model(self):
        helper.save_best(self.Q)

    def load_best_model(self):
        self.Q = helper.load_best()


    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write. 
    #   Function Helper: IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        # print(f"IN helper_func.")
        snake_x, snake_y, body, food_x, food_y = state
        diff_x = snake_x - food_x
        diff_y = snake_y - food_y
        food_dir_x, food_dir_y = 1, 1

        if diff_x > 0:
            food_dir_x = 0
        elif diff_x < 0:
            food_dir_x = 2
        
        if diff_y > 0:
            food_dir_y = 0
        elif diff_y < 0:
            food_dir_y = 2
        
        adj_wall_x, adj_wall_y = 1, 1
        if snake_x == helper.BOARD_LIMIT_MIN:
            adj_wall_x = 0
        elif snake_x == helper.BOARD_LIMIT_MAX:
            adj_wall_x = 2
        
        if snake_y == helper.BOARD_LIMIT_MIN:
            adj_wall_y = 0
        elif snake_y == helper.BOARD_LIMIT_MAX:
            adj_wall_y = 2

        adj_top, adj_bot, adj_left, adj_right = 0, 0, 0, 0
        for body_part in body:
            if body_part[0] == snake_x - helper.GRID_SIZE:
                adj_left = 1
            elif body_part[0] == snake_x + helper.GRID_SIZE:
                adj_right = 1

            # Note that in graphis it's normally quadrant IV
            if body_part[1] == snake_y + helper.GRID_SIZE:
                adj_bot = 1
            elif body_part[1] == snake_y - helper.GRID_SIZE:
                adj_top = 1

        return (adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_top, adj_bot, adj_left, adj_right)

    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    # def update_body(self, body):
        # for i, part in enumerate(body):

    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        # print(f"IN AGENT_ACTION")
        if dead:
            return None

        self.points = points
        wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right = self.helper_func(state)
        samples = np.array([-float('inf')] * len(self.actions))
        future = np.array([-float('inf')] * len(self.actions))
        alpha = .7

        for i in self.actions:
            self.N = self.Q.copy()
            successor = state.copy()
            successor[2] = state[2].copy()
            successor[2].append((state[0], state[1]))
            successor[2].pop(0)

            if i == 0:
                successor[1] -= helper.GRID_SIZE
            elif i == 1:
                successor[1] += helper.GRID_SIZE
            elif i == 2:
                successor[0] -= helper.GRID_SIZE
            else:
                successor[0] += helper.GRID_SIZE
            
            successor_points, successor_dead = self.points, False
            
            if (successor[0], successor[1]) == (successor[3], successor[4]):
                successor_points += 1

            if (successor[0], successor[1]) in state[2] or\
                (successor[0], successor[1]) in successor[2] or\
                successor[0] < helper.BOARD_LIMIT_MIN or\
                successor[0] > helper.BOARD_LIMIT_MAX or\
                successor[1] < helper.BOARD_LIMIT_MIN or\
                successor[1] > helper.BOARD_LIMIT_MAX:
                successor_dead = True

            samples[i] = self.compute_reward(successor_points, successor_dead)
            wx1, wy1, fdx1, fdy1, t1, b1, l1, r1 = self.helper_func(successor)
            succ_max = np.max(self.Q[wx1, wy1, fdx1, fdy1, t1, b1, l1, r1, :])
            samples[i] += self.gamma * succ_max

            # Pretend that we make move i, and see what the N-table looks like
            if not successor_dead:
                nval_old = self.N[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, i]
                self.N[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, i] = (1 - alpha) * nval_old + alpha * samples[i]
                future[i] = self.N[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, i]
        
        max_action = np.argmax(future)

        if self._train:
            qval_old = self.Q[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, max_action]
            self.Q[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, max_action] = (1 - alpha) * qval_old + alpha * samples[max_action]    

        return max_action