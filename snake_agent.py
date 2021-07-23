import numpy as np
import helper
import random

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
        print(f"IN helper_func.")
        snake_x, snake_y, body, food_x, food_y = state
        diff_x = snake_x - food_x
        diff_y = snake_y - food_y
        food_dir_x, food_dir_y = 1, 1

        if diff_x < 0:
            food_dir_x = 0
        elif diff_x > 0:
            food_dir_x = 2
        
        if diff_y < 0:
            food_dir_y = 0
        elif diff_y > 0:
            food_dir_y = 2
        
        adj_wall_x, adj_wall_y = 1, 1
        if snake_x == helper.BOARD_LIMIT_MIN + helper.GRID_SIZE:
            adj_wall_x = 0
        elif snake_x == helper.BOARD_LIMIT_MAX - helper.GRID_SIZE:
            adj_wall_x = 2
        
        if snake_y == helper.BOARD_LIMIT_MIN + helper.GRID_SIZE:
            adj_wall_y = 0
        elif snake_y == helper.BOARD_LIMIT_MAX - helper.GRID_SIZE:
            adj_wall_y = 2

        adj_top, adj_bot, adj_left, adj_right = 0, 0, 0, 0
        for b in body:
            if b[0] == snake_x - helper.GRID_SIZE:
                adj_left = 1
            elif b[0] == snake_x + helper.GRID_SIZE:
                adj_right = 1

            # Note that in graphis it's normally quadrant IV
            if b[1] == snake_y + helper.GRID_SIZE:
                adj_bot = 1
            elif b[1] == snake_y - helper.GRID_SIZE:
                adj_top = 1

        # actions = self.Q[adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_top, adj_bot, adj_left, adj_right, :]
        return (adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_top, adj_bot, adj_left, adj_right)



    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

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
        print(f"IN AGENT_ACTION")
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right = self.helper_func(state)
        action = np.argmax(self.Q[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, :])
        qval_old = self.Q[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, action]
        sample = 0

        self.Q[wall_x, wall_y, food_dir_x, food_dir_y, top, bot, left, right, action] = (1 - 0.7) * qval_old + 0.7 * sample

        new_x_pos, new_x_neg = state[0] + helper.GRID_SIZE, state[0] - helper.GRID_SIZE
        new_y_pos, new_y_neg = state[0] - helper.GRID_SIZE, state[0] + helper.GRID_SIZE

        #UNCOMMENT THIS TO RETURN THE REQUIRED ACTION.
        return action